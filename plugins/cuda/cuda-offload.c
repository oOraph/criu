/*
 * cuda-offload - Offload GPU VRAM to disk, process stays frozen throughout.
 *
 * Checkpoint (run before criu dump):
 *   1. PTRACE_SEIZE + PTRACE_INTERRUPT the whole process tree
 *   2. cuda-checkpoint --action lock     (quiesce CUDA via UVM driver, works while frozen)
 *   3. Snapshot anonymous VMAs (before)
 *   4. Resume ONLY the CUDA restore thread (--get-restore-tid)
 *   5. cuda-checkpoint --action checkpoint (VRAM -> CPU RAM via UVM driver)
 *   6. Re-interrupt the restore thread
 *   7. Diff VMAs before/after, dump new ones to gpu-pages-<pid>.img
 *   8. Inject madvise(MADV_DONTNEED) to free the CPU RAM
 *   9. PTRACE_DETACH with SIGSTOP -> process stays T_STOPPED for criu dump
 *
 * Restore (run after criu restore --leave-stopped):
 *   1. PTRACE_SEIZE + PTRACE_INTERRUPT the process tree (all T_STOPPED from criu)
 *   2. Inject mmap(MAP_FIXED|MAP_SHARED) from image file over GPU VMAs
 *   3. Inject MADV_POPULATE_READ + mlock to fault pages in
 *   4. Resume ONLY the CUDA restore thread
 *   5. cuda-checkpoint --action restore  (pages -> VRAM via UVM driver)
 *   6. cuda-checkpoint --action unlock
 *   7. Re-interrupt the restore thread
 *   8. PTRACE_DETACH in reverse BFS order + SIGCONT -> process tree resumes
 *
 * Timer correctness: wall-clock timers (ITIMER_REAL, POSIX CLOCK_REALTIME/
 * MONOTONIC/BOOTTIME) are saved and disarmed at seize time in op 1, then
 * re-armed at the very end of op 4 just before SIGCONT.  This ensures a timer
 * that had N seconds remaining at the start of op 1 still has ~N seconds
 * remaining when the process wakes up, regardless of how much real time elapsed
 * across ops 1-4.  CPU-time timers (ITIMER_VIRTUAL/PROF, CLOCK_PROCESS_CPUTIME)
 * do not tick while the process is stopped and need no correction.
 *
 * Usage:
 *   cuda-offload --pid PID --dir DIR --action checkpoint
 *   cuda-offload --pid PID --dir DIR --action restore
 *
 *   Restore requires: criu restore --leave-stopped ...
 *
 * By default the tool recurses into the full process subtree rooted at PID.
 * Pass --no-recurse to operate on PID only.
 *
 * Reliability note (multi-process trees + criu restore):
 *   When criu cannot preserve PIDs, restore uses a BFS positional mapping:
 *   gpu-offload-pids.img records the checkpoint BFS order, and the live tree
 *   is traversed in the same order so ckpt_pids[i] -> live_pids[i].
 */

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>

/*
 * PTRACE_GETSIGMASK/SETSIGMASK require the kernel sigset size (8 bytes),
 * not glibc's sigset_t (128 bytes).  EINVAL results if the wrong size is used.
 */
typedef struct { uint64_t sig; } k_sigset_t;

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

#define PID_LIST_FILE   "gpu-offload-pids.img"
/*
 * Marker written by cuda-offload checkpoint. Its presence tells the CRIU
 * plugin to skip restore_gpu_pages during `criu restore` — cuda-offload
 * restore will reload the pages externally.
 */
#define EXTERNAL_MARKER "gpu-offload-external.marker"

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

/*
 * Query the CUDA restore thread TID for a process.
 * Returns the TID on success, -1 if the process has no CUDA context or
 * cuda-checkpoint is unavailable.
 */
static int get_restore_tid(int pid)
{
	char pid_str[32], out[64];
	pid_t child;
	int pfd[2], status;
	ssize_t n;

	snprintf(pid_str, sizeof(pid_str), "%d", pid);

	if (pipe(pfd) < 0) {
		pr_perror("pipe for get-restore-tid");
		return -1;
	}
	child = fork();
	if (child < 0) {
		pr_perror("fork for get-restore-tid");
		close(pfd[0]);
		close(pfd[1]);
		return -1;
	}
	if (child == 0) {
		close(pfd[0]);
		dup2(pfd[1], STDOUT_FILENO);
		close(pfd[1]);
		execlp("cuda-checkpoint", "cuda-checkpoint",
		       "--get-restore-tid", "--pid", pid_str, NULL);
		_exit(1);
	}
	close(pfd[1]);
	n = read(pfd[0], out, sizeof(out) - 1);
	close(pfd[0]);
	if (waitpid(child, &status, 0) < 0 ||
	    !WIFEXITED(status) || WEXITSTATUS(status) != 0)
		return -1;
	if (n <= 0)
		return -1;
	out[n] = '\0';
	return atoi(out);
}

/* ---- ptrace helpers ---- */

/*
 * PTRACE_SEIZE + PTRACE_INTERRUPT: attach to pid without disturbing its state,
 * then force it into ptrace-stop.  On success the caller is the tracer and pid
 * is in ptrace-stop.  Returns 0 on success, -1 on error.
 */
static int ptrace_seize_stop(int pid)
{
	int status;

	if (ptrace(PTRACE_SEIZE, pid, NULL, NULL) < 0) {
		pr_perror("PTRACE_SEIZE failed for pid %d", pid);
		return -1;
	}
	if (ptrace(PTRACE_INTERRUPT, pid, NULL, NULL) < 0) {
		pr_perror("PTRACE_INTERRUPT failed for pid %d", pid);
		ptrace(PTRACE_DETACH, pid, NULL, NULL);
		return -1;
	}
	if (waitpid(pid, &status, __WALL) < 0) {
		pr_perror("waitpid after PTRACE_INTERRUPT failed for pid %d", pid);
		ptrace(PTRACE_DETACH, pid, NULL, NULL);
		return -1;
	}
	return 0;
}

/*
 * PTRACE_DETACH injecting SIGSTOP: the process re-enters T_STOPPED
 * (group-stop) immediately on detach, suitable for hand-off to criu dump.
 */
static void ptrace_detach_stopped(int pid)
{
	ptrace(PTRACE_DETACH, pid, NULL, (void *)(long)SIGSTOP);
}

/*
 * PTRACE_DETACH with no signal: the process resumes running.
 * Use SIGCONT separately for threads still in group-stop.
 */
static void ptrace_detach_resume(int pid)
{
	ptrace(PTRACE_DETACH, pid, NULL, NULL);
}

/*
 * Resume a single thread for CUDA IPC while the rest of the process stays
 * frozen.  Blocks all signals except SIGTRAP to prevent spurious delivery
 * during the brief resume window.  Saves the original sigmask in *save.
 */
static int resume_one_thread(int pid, int tid, k_sigset_t *save)
{
	k_sigset_t block;

	if (ptrace(PTRACE_GETSIGMASK, tid, sizeof(*save), save) < 0) {
		pr_perror("PTRACE_GETSIGMASK failed for tid %d", tid);
		return -1;
	}
	block.sig = ~0ULL & ~(1ULL << (SIGTRAP - 1));
	if (ptrace(PTRACE_SETSIGMASK, tid, sizeof(block), &block) < 0) {
		pr_perror("PTRACE_SETSIGMASK failed for tid %d", tid);
		return -1;
	}
	/* clear PTRACE_O_SUSPEND_SECCOMP so CUDA IPC syscalls are not filtered */
	ptrace(PTRACE_SETOPTIONS, tid, NULL, (void *)0L);
	if (ptrace(PTRACE_CONT, tid, NULL, NULL) < 0) {
		pr_perror("PTRACE_CONT failed for tid %d", tid);
		ptrace(PTRACE_SETSIGMASK, tid, sizeof(*save), save);
		return -1;
	}
	/*
	 * If the process was in group-stop (e.g. CRIU restored it while stopped),
	 * PTRACE_CONT alone does not exit group-stop — the thread re-enters it
	 * immediately.  SIGCONT clears the group-stop for the whole process so the
	 * thread can actually execute.  Threads already in ptrace-stop (main thread
	 * seized by the outer loop) are unaffected by SIGCONT.
	 */
	kill(pid, SIGCONT);
	return 0;
}

/*
 * Re-interrupt a thread after CUDA IPC and restore its original sigmask.
 * Also sets PTRACE_O_SUSPEND_SECCOMP to prevent seccomp interference.
 */
static int interrupt_one_thread(int tid, k_sigset_t *save)
{
	int status;

	if (ptrace(PTRACE_INTERRUPT, tid, NULL, NULL) < 0) {
		pr_perror("PTRACE_INTERRUPT failed for tid %d", tid);
		return -1;
	}
	if (waitpid(tid, &status, __WALL) < 0) {
		pr_perror("waitpid after PTRACE_INTERRUPT failed for tid %d", tid);
		return -1;
	}
	if (ptrace(PTRACE_SETOPTIONS, tid, NULL,
		   (void *)(long)PTRACE_O_SUSPEND_SECCOMP) < 0) {
		/* non-fatal: seccomp may not be active */
	}
	if (ptrace(PTRACE_SETSIGMASK, tid, sizeof(*save), save) < 0) {
		pr_perror("PTRACE_SETSIGMASK failed for tid %d", tid);
		return -1;
	}
	return 0;
}

/* ---- per-pid checkpoint ---- */

/*
 * Checkpoint one process.  The caller must have already PTRACE_SEIZE'd pid
 * (pid is in ptrace-stop).  On return pid is still in ptrace-stop; the caller
 * is responsible for detaching with SIGSTOP when done with the whole tree.
 *
 * Returns 0 on success or if the process has no CUDA context (skipped).
 * Returns -1 on hard errors.
 */
static int do_checkpoint_one(int pid, int img_dir_fd)
{
	struct gpu_region *vmas_before = NULL, *vmas_after = NULL;
	struct gpu_region *new_vmas = NULL;
	int n_before = 0, n_after = 0, n_new = 0;
	double t0, elapsed;
	double total_mb = 0;
	uint64_t syscall_addr;
	k_sigset_t save_sigset;
	int restore_tid;
	int i, ret = 0;

	/*
	 * Check whether this process has a CUDA context before doing anything.
	 * get_restore_tid works while the process is ptrace-stopped because
	 * cuda-checkpoint communicates via the UVM kernel driver, not via
	 * userspace IPC with the target process.
	 */
	restore_tid = get_restore_tid(pid);
	if (restore_tid == -1) {
		pr_info("pid %d: no CUDA context, skipping\n", pid);
		return 0;
	}
	/* 1. Lock CUDA (quiesce the runtime via UVM driver, process stays frozen) */
	t0 = now_ms();
	if (run_cuda_checkpoint(pid, "lock") != 0) {
		pr_info("pid %d: lock failed, skipping\n", pid);
		return 0;
	}
	pr_info("[timing] pid %d lock: %.0f ms\n", pid, now_ms() - t0);

	/* 2. Seize the restore thread so we can briefly resume it for checkpoint */
	if (ptrace_seize_stop(restore_tid) != 0) {
		pr_err("pid %d: failed to seize restore_tid %d\n", pid, restore_tid);
		run_cuda_checkpoint(pid, "unlock");
		return -1;
	}

	/* 3. Pre-scan: anonymous private VMAs before VRAM moves to RAM */
	if (scan_anon_private_vmas(pid, &vmas_before, &n_before) != 0)
		pr_warn("pid %d: pre-scan failed; cannot identify GPU VMAs\n", pid);

	/*
	 * 4. Resume only the restore thread so cuda-checkpoint can drive the
	 * VRAM->RAM transfer.  All other threads (including the main thread)
	 * remain in ptrace-stop throughout.
	 */
	if (resume_one_thread(pid, restore_tid, &save_sigset) != 0) {
		ret = -1;
		goto out_interrupt;
	}

	/* 5. Checkpoint: cuda-checkpoint moves VRAM into new anon VMAs */
	t0 = now_ms();
	if (run_cuda_checkpoint(pid, "checkpoint") != 0) {
		pr_err("pid %d: cuda-checkpoint checkpoint failed\n", pid);
		ret = -1;
	}
	pr_info("[timing] pid %d checkpoint: %.0f ms\n", pid, now_ms() - t0);

out_interrupt:
	/* 6. Re-freeze the restore thread regardless of checkpoint outcome */
	if (interrupt_one_thread(restore_tid, &save_sigset) != 0)
		ret = -1;

	if (ret != 0)
		goto out;

	if (!vmas_before)
		goto out;

	/* 7. Diff VMAs: new ones contain the VRAM data */
	t0 = now_ms();
	if (scan_anon_private_vmas(pid, &vmas_after, &n_after) != 0 ||
	    diff_anon_vmas(vmas_before, n_before, vmas_after, n_after,
			   &new_vmas, &n_new) != 0) {
		pr_warn("pid %d: post-scan/diff failed; skipping GPU page dump\n", pid);
		goto out;
	}
	pr_info("[timing] pid %d post-scan+diff: %.0f ms, %d new VMAs\n",
		pid, now_ms() - t0, n_new);

	if (n_new == 0) {
		pr_warn("pid %d: no new GPU VMAs found — no VRAM to offload\n", pid);
		goto out;
	}

	for (i = 0; i < n_new; i++)
		total_mb += new_vmas[i].size / (1024.0 * 1024.0);
	pr_info("pid %d: found %d GPU VMAs (%.0f MB total)\n",
		pid, n_new, (double)total_mb);

	/* 8. Dump GPU pages to image */
	t0 = now_ms();
	if (dump_gpu_pages(pid, img_dir_fd, new_vmas, n_new) != 0) {
		pr_warn("pid %d: GPU page dump failed; CPU RAM not freed\n", pid);
		goto out;
	}
	elapsed = now_ms() - t0;
	pr_info("[timing] pid %d dump: %.0f ms (%.1f GB/s)\n",
		pid, elapsed, total_mb / elapsed * 1e3 / 1024.0);

	/* 9. Free CPU RAM via injected madvise(MADV_DONTNEED) */
	syscall_addr = find_syscall_addr(pid);
	if (!syscall_addr) {
		pr_warn("pid %d: could not find syscall insn in vdso; CPU RAM not freed\n",
			pid);
		goto out;
	}

	t0 = now_ms();
	if (release_gpu_pages(pid, syscall_addr, new_vmas, n_new) == 0)
		pr_info("[timing] pid %d madvise(DONTNEED): %.0f ms — CPU RAM freed\n",
			pid, now_ms() - t0);
	else
		pr_warn("pid %d: madvise(DONTNEED) failed; CPU RAM not freed\n", pid);

out:
	/*
	 * Detach the restore thread with SIGSTOP so it returns to T_STOPPED.
	 * The main thread (pid) stays in ptrace-stop under the caller's control.
	 */
	ptrace_detach_stopped(restore_tid);
	free(vmas_before);
	free(vmas_after);
	free(new_vmas);
	return ret;
}

/* ---- per-pid restore ---- */

/*
 * Restore one process.  The caller must have already PTRACE_SEIZE'd pid
 * (pid is in ptrace-stop, as left by criu restore --leave-stopped).
 * img_pid is the pid encoded in the image filename.
 *
 * On return pid is still in ptrace-stop; the caller detaches in reverse BFS
 * order and sends SIGCONT to resume the whole tree.
 *
 * Returns 0 on success, -1 on error.
 */
static int do_restore_one(int pid, int img_dir_fd, int img_pid)
{
	k_sigset_t save_sigset;
	uint64_t syscall_addr;
	int restore_tid;
	double t0;

	syscall_addr = find_syscall_addr(pid);
	if (!syscall_addr) {
		pr_err("pid %d: could not find syscall insn in vdso\n", pid);
		return -1;
	}

	/* 1. Remap GPU VMAs from the image file (mmap injection, no CUDA needed) */
	t0 = now_ms();
	if (restore_gpu_pages(pid, img_pid, syscall_addr, img_dir_fd) != 0)
		return -1;
	pr_info("[timing] pid %d mmap+mlock: %.0f ms\n", pid, now_ms() - t0);

	restore_tid = get_restore_tid(pid);
	if (restore_tid == -1) {
		pr_warn("pid %d: no CUDA restore thread found after page restore\n", pid);
		return -1;
	}

	/* 2. Seize the restore thread */
	if (ptrace_seize_stop(restore_tid) != 0) {
		pr_err("pid %d: failed to seize restore_tid %d\n", pid, restore_tid);
		return -1;
	}

	/* 3. Resume only the restore thread for CUDA restore + unlock */
	if (resume_one_thread(pid, restore_tid, &save_sigset) != 0) {
		ptrace_detach_stopped(restore_tid);
		return -1;
	}

	t0 = now_ms();
	if (run_cuda_checkpoint(pid, "restore") != 0) {
		pr_err("pid %d: cuda-checkpoint restore failed\n", pid);
		interrupt_one_thread(restore_tid, &save_sigset);
		ptrace_detach_stopped(restore_tid);
		return -1;
	}
	pr_info("[timing] pid %d restore: %.0f ms\n", pid, now_ms() - t0);

	t0 = now_ms();
	if (run_cuda_checkpoint(pid, "unlock") != 0) {
		pr_err("pid %d: cuda-checkpoint unlock failed\n", pid);
		interrupt_one_thread(restore_tid, &save_sigset);
		ptrace_detach_stopped(restore_tid);
		return -1;
	}
	pr_info("[timing] pid %d unlock: %.0f ms\n", pid, now_ms() - t0);

	/* 4. Re-freeze the restore thread */
	interrupt_one_thread(restore_tid, &save_sigset);

	/*
	 * Detach the restore thread with SIGSTOP so it stays in T_STOPPED until
	 * the caller sends SIGCONT to the whole process.
	 */
	ptrace_detach_stopped(restore_tid);
	return 0;
}

/* ---- timer state save/restore ---- */

#define TIMERS_IMG_PREFIX "gpu-offload-timers-"
#define MAX_POSIX_TIMERS 64

static int peek_word(int pid, uint64_t addr, long *out)
{
	errno = 0;
	*out = ptrace(PTRACE_PEEKDATA, pid, (void *)addr, NULL);
	return errno ? -1 : 0;
}

static int poke_word(int pid, uint64_t addr, long val)
{
	return ptrace(PTRACE_POKEDATA, pid, (void *)addr, (void *)val) < 0 ? -1 : 0;
}

/*
 * Enumerate /proc/<pid>/timers and collect wall-clock POSIX timer IDs
 * (CLOCK_REALTIME=0, CLOCK_MONOTONIC=1, CLOCK_BOOTTIME=7).
 * Returns number of timers found.
 */
static int read_posix_timer_list(int pid, int timer_ids[MAX_POSIX_TIMERS],
				 int clock_ids[MAX_POSIX_TIMERS])
{
	char path[64];
	FILE *f;
	char line[256];
	int cur_id = -1, cur_clk = -1, n = 0;

	snprintf(path, sizeof(path), "/proc/%d/timers", pid);
	f = fopen(path, "r");
	if (!f)
		return 0;

	while (fgets(line, sizeof(line), f)) {
		if (strncmp(line, "ID:", 3) == 0) {
			if (cur_id >= 0 && cur_clk >= 0 &&
			    (cur_clk == 0 || cur_clk == 1 || cur_clk == 7) &&
			    n < MAX_POSIX_TIMERS) {
				timer_ids[n] = cur_id;
				clock_ids[n++] = cur_clk;
			}
			cur_id = atoi(line + 3);
			cur_clk = -1;
		} else if (strncmp(line, "ClockID:", 8) == 0) {
			cur_clk = atoi(line + 8);
		}
	}
	if (cur_id >= 0 && cur_clk >= 0 &&
	    (cur_clk == 0 || cur_clk == 1 || cur_clk == 7) &&
	    n < MAX_POSIX_TIMERS) {
		timer_ids[n] = cur_id;
		clock_ids[n++] = cur_clk;
	}
	fclose(f);
	return n;
}

/*
 * Atomically disarm ITIMER_REAL and retrieve its previous value via:
 *   setitimer(ITIMER_REAL, {0}, old_value)
 * scratch[0..31] must already contain zeros (new_value = disarm).
 * old_value is written to scratch[32..63].
 * Returns 0 on success, -1 on error.
 */
static int atomic_disarm_itimer(int pid, uint64_t syscall_addr, uint64_t scratch,
				long old[4])
{
	int i;

	if (inject_syscall(pid, syscall_addr, __NR_setitimer,
			   0 /* ITIMER_REAL */,
			   (long)scratch,
			   (long)(scratch + 32), 0, 0, 0) < 0)
		return -1;
	for (i = 0; i < 4; i++) {
		if (peek_word(pid, scratch + 32 + (uint64_t)i * 8, &old[i]) < 0)
			return -1;
	}
	return 0;
}

/*
 * Atomically disarm one POSIX timer and retrieve its previous value via:
 *   timer_settime(id, 0, {0}, old_value)
 * scratch[0..31] must already contain zeros (new_value = disarm).
 * old_value is written to scratch[32..63].
 * Returns 0 on success, -1 on error (e.g. timer_id no longer valid).
 */
static int atomic_disarm_posix_timer(int pid, uint64_t syscall_addr,
				     uint64_t scratch, int timer_id, long old[4])
{
	int i;

	if (inject_syscall(pid, syscall_addr, __NR_timer_settime,
			   (long)timer_id, 0,
			   (long)scratch,
			   (long)(scratch + 32), 0, 0) < 0)
		return -1;
	for (i = 0; i < 4; i++) {
		if (peek_word(pid, scratch + 32 + (uint64_t)i * 8, &old[i]) < 0)
			return -1;
	}
	return 0;
}

/*
 * Atomically disarm all wall-clock timers and save their previous values to
 * gpu-offload-timers-<pid>.img (distinct from CRIU's itimers-<pid>.img /
 * posix-timers-<pid>.img — no collision).
 *
 * Using setitimer/timer_settime with old_value argument is atomic: disarm and
 * read are one syscall.  If the timer had already fired (old it_value == {0,0}),
 * no entry is written and the timer is not re-armed at restore.  The pending
 * signal (queued by the kernel before we disarmed) is captured by CRIU and
 * delivered on SIGCONT — correct behaviour since the timer was expiring anyway.
 */
static void save_and_disarm_timers_for_pid(int pid, int img_dir_fd)
{
	uint64_t syscall_addr;
	long scratch_l;
	uint64_t scratch;
	long itimer_old[4];
	long posix_old[MAX_POSIX_TIMERS][4];
	int posix_ok[MAX_POSIX_TIMERS];
	int timer_ids[MAX_POSIX_TIMERS], clock_ids[MAX_POSIX_TIMERS];
	int n_posix, i;
	char name[64];
	int fd;
	FILE *f;

	syscall_addr = find_syscall_addr(pid);
	if (!syscall_addr) {
		pr_warn("pid %d: save_timers: no syscall addr\n", pid);
		return;
	}

	scratch_l = inject_syscall(pid, syscall_addr, __NR_mmap, 0, 4096,
				   PROT_READ | PROT_WRITE,
				   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (scratch_l <= 0) {
		pr_warn("pid %d: save_timers: mmap failed (%ld)\n", pid, scratch_l);
		return;
	}
	scratch = (uint64_t)scratch_l;

	/* Zero scratch[0..31] once — reused as new_value={0} for all disarm calls */
	for (i = 0; i < 4; i++)
		poke_word(pid, scratch + (uint64_t)i * 8, 0);

	/* Atomically disarm ITIMER_REAL and get its old value */
	if (atomic_disarm_itimer(pid, syscall_addr, scratch, itimer_old) < 0) {
		pr_warn("pid %d: save_timers: ITIMER_REAL disarm failed\n", pid);
		goto out_munmap;
	}

	/* Atomically disarm each wall-clock POSIX timer and get its old value */
	n_posix = read_posix_timer_list(pid, timer_ids, clock_ids);
	for (i = 0; i < n_posix; i++)
		posix_ok[i] = (atomic_disarm_posix_timer(pid, syscall_addr, scratch,
							  timer_ids[i],
							  posix_old[i]) == 0);

	/* Write only active timers to the image file */
	snprintf(name, sizeof(name), TIMERS_IMG_PREFIX "%d.img", pid);
	fd = openat(img_dir_fd, name, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (fd < 0) {
		pr_perror("pid %d: save_timers: openat %s", pid, name);
		goto out_munmap;
	}
	f = fdopen(fd, "w");
	if (!f) {
		close(fd);
		goto out_munmap;
	}

	/* itimer_old: [it_interval.sec, it_interval.usec, it_value.sec, it_value.usec] */
	if (itimer_old[2] != 0 || itimer_old[3] != 0)
		fprintf(f, "ITIMER_REAL %ld %ld %ld %ld\n",
			itimer_old[0], itimer_old[1], itimer_old[2], itimer_old[3]);

	for (i = 0; i < n_posix; i++) {
		if (!posix_ok[i])
			continue;
		/* posix_old: [it_interval.sec, it_interval.nsec, it_value.sec, it_value.nsec] */
		if (posix_old[i][2] == 0 && posix_old[i][3] == 0)
			continue;  /* timer had already fired before we disarmed it */
		fprintf(f, "POSIX %d %d %ld %ld %ld %ld\n",
			timer_ids[i], clock_ids[i],
			posix_old[i][0], posix_old[i][1],
			posix_old[i][2], posix_old[i][3]);
	}
	fclose(f);
	pr_info("pid %d: disarmed timers (ITIMER_REAL value=%lds, %d POSIX)\n",
		pid, itimer_old[2], n_posix);
out_munmap:
	inject_syscall(pid, syscall_addr, __NR_munmap, (long)scratch, 4096,
		       0, 0, 0, 0);
}

/*
 * Re-arm timers from gpu-offload-timers-<pid>.img, overriding CRIU's re-arming.
 * Called just before SIGCONT so timers start from checkpoint state, not from
 * the moment CRIU restored them.
 */
static void restore_timers_for_pid(int pid, int img_dir_fd)
{
	char name[64];
	int fd;
	FILE *f;
	uint64_t syscall_addr;
	long scratch_l;
	uint64_t scratch;
	char line[256];

	snprintf(name, sizeof(name), TIMERS_IMG_PREFIX "%d.img", pid);
	fd = openat(img_dir_fd, name, O_RDONLY);
	if (fd < 0)
		return;  /* no saved timer state */
	f = fdopen(fd, "r");
	if (!f) {
		close(fd);
		return;
	}

	syscall_addr = find_syscall_addr(pid);
	if (!syscall_addr) {
		pr_warn("pid %d: restore_timers: no syscall addr\n", pid);
		fclose(f);
		return;
	}

	scratch_l = inject_syscall(pid, syscall_addr, __NR_mmap, 0, 4096,
				   PROT_READ | PROT_WRITE,
				   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (scratch_l <= 0) {
		pr_warn("pid %d: restore_timers: mmap failed\n", pid);
		fclose(f);
		return;
	}
	scratch = (uint64_t)scratch_l;

	while (fgets(line, sizeof(line), f)) {
		if (strncmp(line, "ITIMER_REAL ", 12) == 0) {
			long w[4];
			int i;

			if (sscanf(line + 12, "%ld %ld %ld %ld",
				   &w[0], &w[1], &w[2], &w[3]) != 4)
				continue;
			if (w[2] == 0 && w[3] == 0)
				continue;  /* was not active */
			for (i = 0; i < 4; i++) {
				if (poke_word(pid, scratch + (uint64_t)i * 8, w[i]) < 0)
					goto out;
			}
			if (inject_syscall(pid, syscall_addr, __NR_setitimer,
					   0, (long)scratch, 0, 0, 0, 0) < 0)
				pr_warn("pid %d: restore_timers: setitimer failed\n", pid);
			else
				pr_info("pid %d: re-armed ITIMER_REAL (value=%lds)\n", pid, w[2]);

		} else if (strncmp(line, "POSIX ", 6) == 0) {
			int timer_id, clock_id;
			long ts[4];
			int j;

			if (sscanf(line + 6, "%d %d %ld %ld %ld %ld",
				   &timer_id, &clock_id,
				   &ts[0], &ts[1], &ts[2], &ts[3]) != 6)
				continue;
			if (ts[2] == 0 && ts[3] == 0)
				continue;  /* was not active */
			for (j = 0; j < 4; j++) {
				if (poke_word(pid, scratch + (uint64_t)j * 8, ts[j]) < 0)
					goto out;
			}
			if (inject_syscall(pid, syscall_addr, __NR_timer_settime,
					   (long)timer_id, 0, (long)scratch, 0, 0, 0) < 0)
				pr_warn("pid %d: restore_timers: timer_settime(%d) failed\n",
					pid, timer_id);
		}
	}
out:
	fclose(f);
	inject_syscall(pid, syscall_addr, __NR_munmap, (long)scratch, 4096,
		       0, 0, 0, 0);
}

/* ---- main ---- */

static void usage(const char *prog)
{
	fprintf(stderr,
		"Usage: %s --pid PID --dir DIR --action checkpoint|restore [--no-recurse]\n"
		"\n"
		"  checkpoint  Seize process tree, lock+checkpoint GPU, spill VRAM to\n"
		"              DIR/gpu-pages-PID.img, free CPU RAM.  Leaves process tree\n"
		"              in T_STOPPED for criu dump to take over immediately.\n"
		"              The process is never resumed between cuda-offload and criu.\n"
		"\n"
		"  restore     Reload pages from image, remap into process, restore+unlock\n"
		"              GPU.  Requires the process tree to be in T_STOPPED state\n"
		"              (criu restore --leave-stopped).  Sends SIGCONT at the end.\n"
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
		if (recurse && write_pid_list(img_dir_fd, pids, n_pids) != 0)
			pr_warn("Failed to write pid list; restore may fail if pids change\n");

		/* Seize and stop the entire process tree before any cuda ops */
		for (i = 0; i < n_pids; i++) {
			if (ptrace_seize_stop(pids[i]) != 0) {
				pr_err("Failed to seize pid %d; detaching already-seized\n",
				       pids[i]);
				while (--i >= 0)
					ptrace_detach_resume(pids[i]);
				ret = 1;
				goto done;
			}
		}

		/* Save and disarm wall-clock timers before any CUDA ops */
		for (i = 0; i < n_pids; i++)
			save_and_disarm_timers_for_pid(pids[i], img_dir_fd);

		/* Checkpoint each process while frozen */
		for (i = 0; i < n_pids; i++) {
			if (do_checkpoint_one(pids[i], img_dir_fd) != 0) {
				pr_err("Checkpoint failed for pid %d\n", pids[i]);
				ret = 1;
			}
		}

		if (ret == 0) {
			int mfd = openat(img_dir_fd, EXTERNAL_MARKER,
					 O_WRONLY | O_CREAT | O_TRUNC, 0644);
			if (mfd < 0)
				pr_perror("Cannot create " EXTERNAL_MARKER);
			else
				close(mfd);
		}

		/*
		 * Detach all processes with SIGSTOP (BFS order, parent first).
		 * All processes enter T_STOPPED; criu dump can attach immediately.
		 */
		for (i = 0; i < n_pids; i++)
			ptrace_detach_stopped(pids[i]);

	} else if (strcmp(action, "restore") == 0) {
		int *ckpt_pids = NULL, n_ckpt = 0;
		int have_mapping = 0;

		if (recurse && read_pid_list(img_dir_fd, &ckpt_pids, &n_ckpt) == 0
		    && n_ckpt == n_pids) {
			have_mapping = 1;
			pr_info("Loaded checkpoint pid list (%d entries)\n", n_ckpt);
		}

		/*
		 * Seize the entire tree first.  After criu restore --leave-stopped
		 * all processes are in T_STOPPED; PTRACE_SEIZE attaches without
		 * changing their state, then PTRACE_INTERRUPT converts to ptrace-stop.
		 */
		for (i = 0; i < n_pids; i++) {
			if (ptrace_seize_stop(pids[i]) != 0) {
				pr_err("Failed to seize pid %d; detaching already-seized\n",
				       pids[i]);
				while (--i >= 0)
					ptrace_detach_stopped(pids[i]);
				free(ckpt_pids);
				ret = 1;
				goto done;
			}
		}

		/* Restore each process while the tree stays frozen */
		for (i = 0; i < n_pids; i++) {
			int cur_pid = pids[i];
			int img_pid;

			if (img_exists(cur_pid, img_dir_fd)) {
				img_pid = cur_pid;
			} else if (have_mapping && img_exists(ckpt_pids[i], img_dir_fd)) {
				img_pid = ckpt_pids[i];
			} else {
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

		/*
		 * Re-arm timers with the values saved at seize time (op 1).  CRIU
		 * also re-arms them, but only to the dump-time values; we override
		 * here so the process sees timers as if no time elapsed across ops 1-4.
		 */
		for (i = 0; i < n_pids; i++)
			restore_timers_for_pid(pids[i], img_dir_fd);

		/*
		 * Resume the tree in reverse BFS order (deepest children first so
		 * parents don't race against unready children).  PTRACE_DETACH lets
		 * the main thread of each process run; SIGCONT resumes any threads
		 * still in group-stop from criu --leave-stopped.
		 */
		for (i = n_pids - 1; i >= 0; i--) {
			ptrace_detach_resume(pids[i]);
			kill(pids[i], SIGCONT);
		}

	} else {
		pr_err("Unknown action: %s\n", action);
		usage(argv[0]);
		ret = 1;
	}

done:
	close(img_dir_fd);
	if (recurse)
		free(pids);
	return ret;
}
