/*
 * cuda-offload - Offload GPU VRAM to disk without CRIU
 *
 * --action checkpoint:
 *   1. cuda-checkpoint --action lock     (quiesce CUDA)
 *   2. cuda-checkpoint --action checkpoint (VRAM -> CPU RAM)
 *   3. ptrace-stop process
 *   4. Diff anonymous VMAs before/after, dump new ones to gpu-pages-<pid>.img
 *   5. Inject madvise(MADV_DONTNEED) to free the CPU RAM
 *   6. ptrace-detach
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

/* ---- helpers ---- */

/*
 * Scan dir_fd for a gpu-pages-<pid>.img file and return the pid found in the
 * filename. Returns -1 if not found. Used on restore when the current PID
 * differs from the checkpoint PID (e.g. after criu restore).
 */
static int find_img_pid(int dir_fd)
{
	DIR *dir;
	struct dirent *ent;
	int img_pid = -1;
	int fd;

	fd = dup(dir_fd);
	if (fd < 0)
		return -1;
	dir = fdopendir(fd);
	if (!dir) {
		close(fd);
		return -1;
	}

	while ((ent = readdir(dir)) != NULL) {
		if (sscanf(ent->d_name, "gpu-pages-%d.img", &img_pid) == 1)
			break;
		img_pid = -1;
	}
	closedir(dir);
	return img_pid;
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

/* ---- main ---- */

static void usage(const char *prog)
{
	fprintf(stderr,
		"Usage: %s --pid PID --dir DIR --action checkpoint|restore\n"
		"\n"
		"  checkpoint  lock+checkpoint GPU, spill VRAM to DIR/gpu-pages-PID.img,\n"
		"              free CPU RAM. GPU stays frozen.\n"
		"  restore     reload pages from image, remap into process,\n"
		"              restore+unlock GPU.\n",
		prog);
}

int main(int argc, char **argv)
{
	int pid = 0;
	const char *dir = NULL, *action = NULL;
	int i;

	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--pid") == 0 && i + 1 < argc)
			pid = atoi(argv[++i]);
		else if (strcmp(argv[i], "--dir") == 0 && i + 1 < argc)
			dir = argv[++i];
		else if (strcmp(argv[i], "--action") == 0 && i + 1 < argc)
			action = argv[++i];
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

	if (strcmp(action, "checkpoint") == 0) {
		struct gpu_region *vmas_before = NULL, *vmas_after = NULL;
		struct gpu_region *new_vmas = NULL;
		int n_before = 0, n_after = 0, n_new = 0;
		double t0, elapsed;
		double total_mb = 0;
		uint64_t syscall_addr;
		int img_dir_fd;

		/* 1. Quiesce CUDA so the process can be checkpointed */
		t0 = now_ms();
		if (run_cuda_checkpoint(pid, "lock") != 0)
			return 1;
		pr_info("[timing] lock: %.0f ms\n", now_ms() - t0);

		/* 2. Snapshot anonymous private VMAs before VRAM moves to RAM */
		if (scan_anon_private_vmas(pid, &vmas_before, &n_before) != 0)
			pr_warn("pre-scan failed; cannot identify GPU VMAs\n");

		/* 3. Checkpoint: cuda-checkpoint moves VRAM into new anon VMAs */
		t0 = now_ms();
		if (run_cuda_checkpoint(pid, "checkpoint") != 0) {
			free(vmas_before);
			return 1;
		}
		pr_info("[timing] checkpoint: %.0f ms\n", now_ms() - t0);

		/* 4. Stop the process so we can inject safely */
		if (ptrace_stop(pid) != 0) {
			free(vmas_before);
			return 1;
		}

		if (!vmas_before)
			goto checkpoint_done;

		/* 5. Diff VMAs: new ones contain the VRAM data */
		t0 = now_ms();
		if (scan_anon_private_vmas(pid, &vmas_after, &n_after) != 0 ||
		    diff_anon_vmas(vmas_before, n_before, vmas_after, n_after,
				   &new_vmas, &n_new) != 0) {
			pr_warn("post-scan/diff failed; skipping GPU page dump\n");
			goto checkpoint_done;
		}
		pr_info("[timing] post-scan+diff: %.0f ms, %d new VMAs\n",
			now_ms() - t0, n_new);

		if (n_new == 0) {
			pr_warn("No new GPU VMAs found — no VRAM to offload\n");
			goto checkpoint_done;
		}

		for (i = 0; i < n_new; i++)
			total_mb += new_vmas[i].size / (1024.0 * 1024.0);
		pr_info("Found %d GPU VMAs (%.0f MB total)\n", n_new, (double)total_mb);

		/* 6. Open the image directory and dump GPU pages */
		img_dir_fd = open(dir, O_RDONLY | O_DIRECTORY);
		if (img_dir_fd < 0) {
			pr_perror("Cannot open dir %s", dir);
			goto checkpoint_done;
		}

		t0 = now_ms();
		if (dump_gpu_pages(pid, img_dir_fd, new_vmas, n_new) != 0) {
			pr_warn("dump failed; CPU RAM not freed\n");
			close(img_dir_fd);
			goto checkpoint_done;
		}
		elapsed = now_ms() - t0;
		pr_info("[timing] dump: %.0f ms (%.1f GB/s)\n",
			elapsed, total_mb / elapsed * 1e3 / 1024.0);
		close(img_dir_fd);

		/* 7. Free the CPU RAM pages via injected madvise(MADV_DONTNEED) */
		syscall_addr = find_syscall_addr(pid);
		if (!syscall_addr) {
			pr_warn("Could not find syscall insn in vdso; CPU RAM not freed\n");
			goto checkpoint_done;
		}

		t0 = now_ms();
		if (release_gpu_pages(pid, syscall_addr, new_vmas, n_new) == 0)
			pr_info("[timing] madvise(DONTNEED): %.0f ms — CPU RAM freed\n",
				now_ms() - t0);
		else
			pr_warn("madvise(DONTNEED) failed; CPU RAM not freed\n");

checkpoint_done:
		ptrace_resume(pid);
		free(vmas_before);
		free(vmas_after);
		free(new_vmas);
		return 0;

	} else if (strcmp(action, "restore") == 0) {
		double t0;
		uint64_t syscall_addr;
		int img_dir_fd;

		/* 1. Stop the process to inject safely */
		if (ptrace_stop(pid) != 0)
			return 1;

		syscall_addr = find_syscall_addr(pid);
		if (!syscall_addr) {
			pr_err("Could not find syscall insn in vdso\n");
			ptrace_resume(pid);
			return 1;
		}

		img_dir_fd = open(dir, O_RDONLY | O_DIRECTORY);
		if (img_dir_fd < 0) {
			pr_perror("Cannot open dir %s", dir);
			ptrace_resume(pid);
			return 1;
		}

		/* 2. Remap GPU VMAs from the image file (zero-copy page-table remap).
		 * The image filename uses the checkpoint PID which may differ from
		 * the current PID after criu restore — scan the dir to find it. */
		int img_pid = find_img_pid(img_dir_fd);
		if (img_pid < 0) {
			pr_err("No gpu-pages-*.img found in %s\n", dir);
			close(img_dir_fd);
			ptrace_resume(pid);
			return 1;
		}
		t0 = now_ms();
		if (restore_gpu_pages(pid, img_pid, syscall_addr, img_dir_fd) != 0) {
			close(img_dir_fd);
			ptrace_resume(pid);
			return 1;
		}
		pr_info("[timing] mmap+mlock: %.0f ms\n", now_ms() - t0);
		close(img_dir_fd);

		ptrace_resume(pid);

		/* 3. Let cuda-checkpoint read the pages back to VRAM */
		t0 = now_ms();
		if (run_cuda_checkpoint(pid, "restore") != 0)
			return 1;
		pr_info("[timing] restore: %.0f ms\n", now_ms() - t0);

		/* 4. Unlock: CUDA API calls allowed again */
		t0 = now_ms();
		if (run_cuda_checkpoint(pid, "unlock") != 0)
			return 1;
		pr_info("[timing] unlock: %.0f ms\n", now_ms() - t0);

		return 0;

	} else {
		pr_err("Unknown action: %s\n", action);
		usage(argv[0]);
		return 1;
	}
}
