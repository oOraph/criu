/*
 * cuda-offload - Offload GPU VRAM to disk without CRIU
 *
 * --action checkpoint:
 *   1. cuda-checkpoint --action lock     (quiesce CUDA)
 *   2. ptrace-stop process               (freeze before pre-scan, closes TOCTOU race)
 *   3. Snapshot anonymous VMAs (before)
 *   4. cuda-checkpoint --action checkpoint (VRAM -> CPU RAM, runs via UVM driver)
 *   5. Diff anonymous VMAs before/after, dump new ones to gpu-pages-<pid>.img
 *   6. Inject madvise(MADV_DONTNEED) to free the CPU RAM
 *   7. ptrace-detach
 *   -> GPU stays frozen, data on disk, zero CPU RAM used.
 *
 * --action restore:
 *   1. ptrace-stop process
 *   2. Inject mmap(MAP_FIXED|MAP_SHARED) from image file over GPU VMAs
 *   3. Inject MADV_POPULATE_READ + mlock to fault pages in
 *   4. ptrace-detach
 *   5. cuda-checkpoint --action restore  (pages -> VRAM)
 *   6. cuda-checkpoint --action unlock
 *
 * Usage:
 *   cuda-offload --pid PID --dir DIR --action checkpoint
 *   cuda-offload --pid PID --dir DIR --action restore
 *
 * By default the tool recurses into the full process subtree rooted at PID.
 * Pass --no-recurse to operate on PID only.
 *
 * Reliability note (multi-process trees + criu restore):
 *   When criu cannot preserve PIDs, restore uses a BFS positional mapping:
 *   gpu-offload-pids.img records the checkpoint BFS order, and the live tree
 *   is traversed in the same order so ckpt_pids[i] -> live_pids[i].
 *   This mapping is only valid if the process tree is structurally identical
 *   at the time cuda-offload restore runs.  Non-GPU children are not locked
 *   by cuda-checkpoint and may exit freely after criu restore, breaking the
 *   correspondence.
 *   To guarantee a stable tree, restore the process into a frozen cgroup v2
 *   (echo 1 > cgroup.freeze) before running criu restore.  Any process
 *   created inside a frozen cgroup is immediately frozen by the kernel, so
 *   the entire tree stays suspended until cuda-offload restore completes and
 *   the orchestrator unfreezes the cgroup.
 */

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <unistd.h>

#define pr_info(fmt, ...)   fprintf(stderr, "cuda-offload: " fmt, ##__VA_ARGS__)
#define pr_warn(fmt, ...)   fprintf(stderr, "cuda-offload: WARNING: " fmt, ##__VA_ARGS__)
#define pr_err(fmt, ...)    fprintf(stderr, "cuda-offload: ERROR: " fmt, ##__VA_ARGS__)
#define pr_perror(fmt, ...) fprintf(stderr, "cuda-offload: ERROR: " fmt ": %s\n", \
				    ##__VA_ARGS__, strerror(errno))

#include "cuda_gpu_pages.h"

/* ---- process tree helpers ---- */

static int append_pid(int **pids, int *n, int *cap, int pid)
{
	if (*n >= *cap) {
		int new_cap = *cap ? *cap * 2 : 16;
		int *tmp = realloc(*pids, (size_t)new_cap * sizeof(int));

		if (!tmp)
			return -1;
		*pids = tmp;
		*cap = new_cap;
	}
	(*pids)[(*n)++] = pid;
	return 0;
}

/*
 * Collect all PIDs in the process subtree rooted at root_pid using
 * /proc/<pid>/task/<tid>/children (BFS).  The root is always first.
 * Returns 0 on success; the caller must free(*out_pids).
 */
static int collect_pids(int root_pid, int **out_pids, int *out_n)
{
	int *pids = NULL, n = 0, cap = 0;
	int i;

	if (append_pid(&pids, &n, &cap, root_pid) < 0)
		return -1;

	for (i = 0; i < n; i++) {
		int pid = pids[i];
		char task_path[64];
		DIR *task_dir;
		struct dirent *tid_ent;

		snprintf(task_path, sizeof(task_path), "/proc/%d/task", pid);
		task_dir = opendir(task_path);
		if (!task_dir)
			continue; /* process may have exited */

		while ((tid_ent = readdir(task_dir)) != NULL) {
			char children_path[320];
			FILE *f;
			int child_pid;

			if (tid_ent->d_name[0] == '.')
				continue;

			snprintf(children_path, sizeof(children_path),
				 "/proc/%d/task/%s/children",
				 pid, tid_ent->d_name);
			f = fopen(children_path, "r");
			if (!f)
				continue;

			while (fscanf(f, "%d", &child_pid) == 1) {
				if (append_pid(&pids, &n, &cap, child_pid) < 0) {
					fclose(f);
					closedir(task_dir);
					free(pids);
					return -1;
				}
			}
			fclose(f);
		}
		closedir(task_dir);
	}

	*out_pids = pids;
	*out_n = n;
	return 0;
}

/* ---- image helpers ---- */

/*
 * Name of the file that stores the BFS-ordered checkpoint pid list.
 * Written at checkpoint time and read at restore time to map checkpoint pids
 * to live pids when they differ (e.g. after criu restore).
 */
#define PID_LIST_FILE "gpu-offload-pids.img"

/*
 * Return 1 if gpu-pages-<pid>.img exists in dir_fd, 0 otherwise.
 */
static int img_exists(int pid, int dir_fd)
{
	char name[64];
	int fd;

	snprintf(name, sizeof(name), "gpu-pages-%d.img", pid);
	fd = openat(dir_fd, name, O_RDONLY);
	if (fd < 0)
		return 0;
	close(fd);
	return 1;
}

/*
 * Return the innermost namespace PID for 'pid' by reading NSpid from
 * /proc/<pid>/status.  Falls back to 'pid' if the info is unavailable.
 * gpu-pages images are keyed on the ns-pid (see dump_gpu_pages in
 * cuda_gpu_pages.c), so we need this when restoring from the host namespace
 * where collect_pids() returns host PIDs but the image filename uses the
 * container-side (ns) PID.
 */
static int get_ns_pid_for_pid(int pid)
{
	char path[64];
	FILE *f;
	char line[256];
	int ns_pid = pid;

	snprintf(path, sizeof(path), "/proc/%d/status", pid);
	f = fopen(path, "r");
	if (!f)
		return pid;
	while (fgets(line, sizeof(line), f)) {
		if (strncmp(line, "NSpid:", 6) == 0) {
			int val, last = pid, consumed;
			char *p = line + 6;

			while (sscanf(p, " %d%n", &val, &consumed) == 1) {
				last = val;
				p += consumed;
			}
			ns_pid = last;
			break;
		}
	}
	fclose(f);
	return ns_pid;
}

/*
 * Write the ordered pid list to PID_LIST_FILE so restore can map checkpoint
 * pids to live pids when they differ after criu restore.
 */
static int write_pid_list(int img_dir_fd, int *pids, int n)
{
	int fd;
	FILE *f;
	int i;

	fd = openat(img_dir_fd, PID_LIST_FILE,
		    O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (fd < 0) {
		pr_perror("Cannot create " PID_LIST_FILE);
		return -1;
	}
	f = fdopen(fd, "w");
	if (!f) {
		close(fd);
		return -1;
	}
	for (i = 0; i < n; i++)
		fprintf(f, "%d\n", pids[i]);
	fclose(f);
	return 0;
}

/*
 * Read the checkpoint pid list saved by write_pid_list.
 * Returns 0 and sets *out_pids / *out_n on success; *out_n == 0 if file absent.
 * Caller must free(*out_pids).
 */
static int read_pid_list(int img_dir_fd, int **out_pids, int *out_n)
{
	int fd, pid, cap = 0;
	int *pids = NULL;
	FILE *f;

	*out_pids = NULL;
	*out_n = 0;

	fd = openat(img_dir_fd, PID_LIST_FILE, O_RDONLY);
	if (fd < 0) {
		if (errno == ENOENT)
			return 0;
		pr_perror("Cannot open " PID_LIST_FILE);
		return -1;
	}
	f = fdopen(fd, "r");
	if (!f) {
		close(fd);
		return -1;
	}
	while (fscanf(f, "%d", &pid) == 1) {
		if (append_pid(&pids, out_n, &cap, pid) < 0) {
			fclose(f);
			free(pids);
			return -1;
		}
	}
	fclose(f);
	*out_pids = pids;
	return 0;
}

/* ---- cuda-checkpoint runner ---- */

static int run_cuda_checkpoint(int pid, const char *action)
{
	char pid_str[32];
	pid_t child;
	int status;

	snprintf(pid_str, sizeof(pid_str), "%d", pid);

	child = fork();
	if (child < 0) {
		pr_perror("fork failed");
		return -1;
	}
	if (child == 0) {
		execlp("cuda-checkpoint", "cuda-checkpoint",
		       "--action", action, "--pid", pid_str, NULL);
		fprintf(stderr, "cuda-offload: execlp cuda-checkpoint: %s\n",
			strerror(errno));
		_exit(1);
	}

	if (waitpid(child, &status, 0) < 0) {
		pr_perror("waitpid for cuda-checkpoint failed");
		return -1;
	}
	if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
		pr_err("cuda-checkpoint --action %s failed (exit %d)\n",
		       action, WIFEXITED(status) ? WEXITSTATUS(status) : -1);
		return -1;
	}
	return 0;
}

/* ---- ptrace helpers ---- */

static int ptrace_stop(int pid)
{
	int status;

	if (ptrace(PTRACE_ATTACH, pid, NULL, NULL) < 0) {
		pr_perror("PTRACE_ATTACH failed for pid %d", pid);
		return -1;
	}
	if (waitpid(pid, &status, __WALL) < 0) {
		pr_perror("waitpid after PTRACE_ATTACH failed for pid %d", pid);
		ptrace(PTRACE_DETACH, pid, NULL, NULL);
		return -1;
	}
	return 0;
}

static void ptrace_resume(int pid)
{
	ptrace(PTRACE_DETACH, pid, NULL, NULL);
}

/* ---- per-pid checkpoint / restore ---- */

/*
 * Checkpoint one process: lock+checkpoint CUDA, dump GPU pages, free CPU RAM.
 * Returns 0 on success, -1 on error.  A non-zero return from cuda-checkpoint
 * itself (e.g. no CUDA context) is treated as a skip, not an error.
 */
static int do_checkpoint_one(int pid, int img_dir_fd)
{
	struct gpu_region *vmas_before = NULL, *vmas_after = NULL;
	struct gpu_region *new_vmas = NULL;
	int n_before = 0, n_after = 0, n_new = 0;
	double t0, elapsed;
	double total_mb = 0;
	uint64_t syscall_addr;
	int i, ret = 0;

	/* 1. Quiesce CUDA */
	t0 = now_ms();
	if (run_cuda_checkpoint(pid, "lock") != 0) {
		pr_info("pid %d: lock failed, skipping (no CUDA context?)\n", pid);
		return 0;
	}
	pr_info("[timing] pid %d lock: %.0f ms\n", pid, now_ms() - t0);

	/*
	 * 2. Freeze the process now that CUDA is locked.  cuda-checkpoint
	 *    communicates with the CUDA runtime via the UVM kernel driver, so
	 *    checkpoint can run with CPU threads stopped.  Freezing here closes
	 *    the TOCTOU window: any anonymous VMA created between pre-scan and
	 *    ptrace_stop would appear as a false-positive GPU VMA and get
	 *    MADV_DONTNEED'd, silently zeroing live application memory.
	 */
	if (ptrace_stop(pid) != 0) {
		free(vmas_before);
		return -1;
	}

	/* 3. Snapshot anonymous private VMAs before VRAM moves to RAM */
	if (scan_anon_private_vmas(pid, &vmas_before, &n_before) != 0)
		pr_warn("pid %d: pre-scan failed; cannot identify GPU VMAs\n", pid);

	/* 4. Checkpoint: cuda-checkpoint moves VRAM into new anon VMAs */
	t0 = now_ms();
	if (run_cuda_checkpoint(pid, "checkpoint") != 0) {
		ptrace_resume(pid);
		free(vmas_before);
		return -1;
	}
	pr_info("[timing] pid %d checkpoint: %.0f ms\n", pid, now_ms() - t0);

	if (!vmas_before)
		goto done;

	/* 5. Diff VMAs: new ones contain the VRAM data */
	t0 = now_ms();
	if (scan_anon_private_vmas(pid, &vmas_after, &n_after) != 0 ||
	    diff_anon_vmas(vmas_before, n_before, vmas_after, n_after,
			   &new_vmas, &n_new) != 0) {
		pr_warn("pid %d: post-scan/diff failed; skipping GPU page dump\n", pid);
		goto done;
	}
	pr_info("[timing] pid %d post-scan+diff: %.0f ms, %d new VMAs\n",
		pid, now_ms() - t0, n_new);

	if (n_new == 0) {
		pr_warn("pid %d: no new GPU VMAs found — no VRAM to offload\n", pid);
		goto done;
	}

	for (i = 0; i < n_new; i++)
		total_mb += new_vmas[i].size / (1024.0 * 1024.0);
	pr_info("pid %d: found %d GPU VMAs (%.0f MB total)\n",
		pid, n_new, (double)total_mb);

	/* 6. Dump GPU pages */
	t0 = now_ms();
	if (dump_gpu_pages(pid, img_dir_fd, new_vmas, n_new) != 0) {
		pr_warn("pid %d: dump failed; CPU RAM not freed\n", pid);
		goto done;
	}
	elapsed = now_ms() - t0;
	pr_info("[timing] pid %d dump: %.0f ms (%.1f GB/s)\n",
		pid, elapsed, total_mb / elapsed * 1e3 / 1024.0);

	/* 7. Free the CPU RAM pages via injected madvise(MADV_DONTNEED) */
	syscall_addr = find_syscall_addr(pid);
	if (!syscall_addr) {
		pr_warn("pid %d: could not find syscall insn in vdso; CPU RAM not freed\n",
			pid);
		goto done;
	}

	t0 = now_ms();
	if (release_gpu_pages(pid, syscall_addr, new_vmas, n_new) == 0)
		pr_info("[timing] pid %d madvise(DONTNEED): %.0f ms — CPU RAM freed\n",
			pid, now_ms() - t0);
	else
		pr_warn("pid %d: madvise(DONTNEED) failed; CPU RAM not freed\n", pid);

done:
	ptrace_resume(pid);
	free(vmas_before);
	free(vmas_after);
	free(new_vmas);
	return ret;
}

/*
 * Restore one process: remap GPU pages from image, restore+unlock CUDA.
 * img_pid is the pid encoded in the image filename (may differ from pid after
 * criu restore).
 */
static int do_restore_one(int pid, int img_dir_fd, int img_pid)
{
	double t0;
	uint64_t syscall_addr;

	/* 1. Stop the process to inject safely */
	if (ptrace_stop(pid) != 0)
		return -1;

	syscall_addr = find_syscall_addr(pid);
	if (!syscall_addr) {
		pr_err("pid %d: could not find syscall insn in vdso\n", pid);
		ptrace_resume(pid);
		return -1;
	}

	/* 2. Remap GPU VMAs from the image file */
	t0 = now_ms();
	if (restore_gpu_pages(pid, img_pid, syscall_addr, img_dir_fd) != 0) {
		ptrace_resume(pid);
		return -1;
	}
	pr_info("[timing] pid %d mmap+mlock: %.0f ms\n", pid, now_ms() - t0);

	ptrace_resume(pid);

	/* 3. Let cuda-checkpoint read the pages back to VRAM */
	t0 = now_ms();
	if (run_cuda_checkpoint(pid, "restore") != 0)
		return -1;
	pr_info("[timing] pid %d restore: %.0f ms\n", pid, now_ms() - t0);

	/* 4. Unlock: CUDA API calls allowed again */
	t0 = now_ms();
	if (run_cuda_checkpoint(pid, "unlock") != 0)
		return -1;
	pr_info("[timing] pid %d unlock: %.0f ms\n", pid, now_ms() - t0);

	return 0;
}

/* ---- main ---- */

static void usage(const char *prog)
{
	fprintf(stderr,
		"Usage: %s --pid PID --dir DIR --action checkpoint|restore [--no-recurse]\n"
		"\n"
		"  checkpoint  lock+checkpoint GPU, spill VRAM to DIR/gpu-pages-PID.img,\n"
		"              free CPU RAM. GPU stays frozen.\n"
		"  restore     reload pages from image, remap into process,\n"
		"              restore+unlock GPU.\n"
		"\n"
		"  By default all processes in the subtree rooted at PID are handled.\n"
		"  Use --no-recurse to operate on PID only.\n",
		prog);
}

int main(int argc, char **argv)
{
	int pid = 0, recurse = 1;
	const char *dir = NULL, *action = NULL;
	int *pids = NULL, n_pids = 0;
	int img_dir_fd = -1;
	int ret = 0, i;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--pid") == 0 && i + 1 < argc)
			pid = atoi(argv[++i]);
		else if (strcmp(argv[i], "--dir") == 0 && i + 1 < argc)
			dir = argv[++i];
		else if (strcmp(argv[i], "--action") == 0 && i + 1 < argc)
			action = argv[++i];
		else if (strcmp(argv[i], "--no-recurse") == 0)
			recurse = 0;
		else {
			fprintf(stderr, "Unknown argument: %s\n", argv[i]);
			usage(argv[0]);
			return 1;
		}
	}

	if (!pid || !dir || !action) {
		usage(argv[0]);
		return 1;
	}

	img_dir_fd = open(dir, O_RDONLY | O_DIRECTORY);
	if (img_dir_fd < 0) {
		pr_perror("Cannot open dir %s", dir);
		return 1;
	}

	/* Build the list of PIDs to operate on */
	if (recurse) {
		if (collect_pids(pid, &pids, &n_pids) < 0) {
			pr_perror("Failed to collect process tree for pid %d", pid);
			close(img_dir_fd);
			return 1;
		}
		pr_info("Found %d process(es) in subtree of pid %d\n", n_pids, pid);
	} else {
		pids = &pid;
		n_pids = 1;
	}

	if (strcmp(action, "checkpoint") == 0) {
		/*
		 * Save the BFS-ordered pid list so that restore can map
		 * checkpoint pids to live pids when they differ after criu restore.
		 */
		if (recurse && write_pid_list(img_dir_fd, pids, n_pids) != 0)
			pr_warn("Failed to write pid list; restore may not work if pids change\n");

		for (i = 0; i < n_pids; i++) {
			if (do_checkpoint_one(pids[i], img_dir_fd) != 0) {
				pr_err("Checkpoint failed for pid %d\n", pids[i]);
				ret = 1;
			}
		}

	} else if (strcmp(action, "restore") == 0) {
		/*
		 * Build a checkpoint_pid → live_pid mapping for the case where
		 * pids changed after criu restore.
		 *
		 * At checkpoint time we saved the BFS-ordered pid list.  At
		 * restore time collect_pids() visits the live tree in the same
		 * BFS order (CRIU preserves tree structure and child ordering),
		 * so ckpt_pids[i] maps to pids[i].
		 */
		int *ckpt_pids = NULL, n_ckpt = 0;
		int have_mapping = 0;

		if (recurse && read_pid_list(img_dir_fd, &ckpt_pids, &n_ckpt) == 0
		    && n_ckpt == n_pids) {
			have_mapping = 1;
			pr_info("Loaded checkpoint pid list (%d entries)\n", n_ckpt);
		}

		for (i = 0; i < n_pids; i++) {
			int cur_pid = pids[i];
			int img_pid;

			/* Prefer direct match (pids unchanged, or no-recurse path). */
			if (img_exists(cur_pid, img_dir_fd)) {
				img_pid = cur_pid;
			} else if (have_mapping && img_exists(ckpt_pids[i], img_dir_fd)) {
				/* Map via saved BFS order (pids changed after criu restore). */
				img_pid = ckpt_pids[i];
			} else {
				/*
				 * Images are keyed on the innermost ns-pid (see
				 * dump_gpu_pages).  When running from the host namespace
				 * collect_pids() returns host PIDs, which differ from the
				 * ns-pid used to name the file.  Try the ns-pid as a
				 * last resort before giving up.
				 */
				int ns_pid = get_ns_pid_for_pid(cur_pid);

				if (ns_pid != cur_pid && img_exists(ns_pid, img_dir_fd)) {
					img_pid = cur_pid;
				} else {
					pr_info("pid %d (ns_pid=%d): no image found, skipping\n",
						cur_pid, ns_pid);
					continue;
				}
			}

			if (do_restore_one(cur_pid, img_dir_fd, img_pid) != 0) {
				pr_err("Restore failed for pid %d\n", cur_pid);
				ret = 1;
			}
		}

		free(ckpt_pids);

	} else {
		pr_err("Unknown action: %s\n", action);
		usage(argv[0]);
		ret = 1;
	}

	close(img_dir_fd);
	if (recurse)
		free(pids);
	return ret;
}
