#include "criu-log.h"
#include "plugin.h"
#include "util.h"
#include "cr_options.h"
#include "pid.h"
#include "proc_parse.h"
#include "seize.h"
#include "fault-injection.h"

#include <common/list.h>
#include <compel/infect.h>

#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef SYS_memfd_create
#define SYS_memfd_create 319
#endif

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC 1U
#endif

#include "cuda_gpu_pages.h"

/* cuda-checkpoint binary should live in your PATH */
#define CUDA_CHECKPOINT "cuda-checkpoint"

/* cuda-checkpoint --action flags */
#define ACTION_LOCK	  "lock"
#define ACTION_CHECKPOINT "checkpoint"
#define ACTION_RESTORE	  "restore"
#define ACTION_UNLOCK	  "unlock"

typedef enum {
	CUDA_TASK_RUNNING = 0,
	CUDA_TASK_LOCKED,
	CUDA_TASK_CHECKPOINTED,
	CUDA_TASK_UNKNOWN = -1
} cuda_task_state_t;

#define CUDA_CKPT_BUF_SIZE (128)

#ifdef LOG_PREFIX
#undef LOG_PREFIX
#endif
#define LOG_PREFIX "cuda_plugin: "

/* Disable plugin functionality if cuda-checkpoint is not in $PATH or driver
 * version doesn't support --action flag
 */
bool plugin_disabled = false;

bool plugin_added_to_inventory = false;

struct pid_info {
	int pid;
	char checkpointed;
	cuda_task_state_t initial_task_state;
	struct list_head list;
};

/* Used to track which PID's we've paused CUDA operations on so far so we can
 * release them after we're done with the DUMP
 */
static LIST_HEAD(cuda_pids);



static void dealloc_pid_buffer(struct list_head *pid_buf)
{
	struct pid_info *info;
	struct pid_info *n;

	list_for_each_entry_safe(info, n, pid_buf, list) {
		list_del(&info->list);
		xfree(info);
	}
}

static int add_pid_to_buf(struct list_head *pid_buf, int pid, cuda_task_state_t state)
{
	struct pid_info *new = xmalloc(sizeof(*new));

	if (new == NULL) {
		return -1;
	}

	new->pid = pid;
	new->checkpointed = 0;
	new->initial_task_state = state;
	list_add_tail(&new->list, pid_buf);

	return 0;
}

static int launch_cuda_checkpoint(const char **args, char *buf, int buf_size)
{
#define READ  0
#define WRITE 1
	int fd[2], buf_off;

	if (pipe(fd) != 0) {
		pr_perror("Couldn't create pipes for reading cuda-checkpoint output");
		return -1;
	}

	buf[0] = '\0';

	int child_pid = fork();
	if (child_pid == -1) {
		pr_perror("Failed to fork to exec cuda-checkpoint");
		close(fd[READ]);
		close(fd[WRITE]);
		return -1;
	}

	if (child_pid == 0) { // child
		if (dup2(fd[WRITE], STDOUT_FILENO) == -1) {
			pr_perror("unable to clone fd %d->%d", fd[WRITE], STDOUT_FILENO);
			_exit(EXIT_FAILURE);
		}
		if (dup2(fd[WRITE], STDERR_FILENO) == -1) {
			pr_perror("unable to clone fd %d->%d", fd[WRITE], STDERR_FILENO);
			_exit(EXIT_FAILURE);
		}
		close(fd[READ]);

		close_fds(STDERR_FILENO + 1);

		execvp(args[0], (char **)args);

		/* We can't use pr_error() as log file fd is closed. */
		fprintf(stderr, "execvp(\"%s\") failed: %s\n", args[0], strerror(errno));

		_exit(EXIT_FAILURE);
	}

	close(fd[WRITE]);
	buf_off = 0;
	/* Reserve one byte for the null charracter. */
	buf_size--;
	while (buf_off < buf_size) {
		int bytes_read;
		bytes_read = read(fd[READ], buf + buf_off, buf_size - buf_off);
		if (bytes_read == -1) {
			pr_perror("Unable to read output of cuda-checkpoint");
			goto err;
		}
		if (bytes_read == 0)
			break;
		buf_off += bytes_read;
	}
	buf[buf_off] = '\0';

	/* Clear out any of the remaining output in the pipe in case the buffer wasn't large enough */
	while (true) {
		char scratch[1024];
		int bytes_read;
		bytes_read = read(fd[READ], scratch, sizeof(scratch));
		if (bytes_read == -1) {
			pr_perror("Unable to read output of cuda-checkpoint");
			goto err;
		}
		if (bytes_read == 0)
			break;
	}
	close(fd[READ]);

	int status, exit_code = -1;
	if (waitpid(child_pid, &status, 0) == -1) {
		pr_perror("Unable to wait for the cuda-checkpoint process %d", child_pid);
		goto err;
	}
	if (WIFSIGNALED(status)) {
		int sig = WTERMSIG(status);
		pr_err("cuda-checkpoint unexpectedly signaled with %d: %s\n", sig, strsignal(sig));
	} else if (WIFEXITED(status)) {
		exit_code = WEXITSTATUS(status);
	} else {
		pr_err("cuda-checkpoint exited improperly: %u\n", status);
	}

	if (exit_code != EXIT_SUCCESS)
		pr_debug("cuda-checkpoint output ===>\n%s\n"
			 "<=== cuda-checkpoint output\n",
			 buf);

	return exit_code;
err:
	kill(child_pid, SIGKILL);
	waitpid(child_pid, NULL, 0);
	return -1;
}

/**
 * Checks if a given flag is supported by the cuda-checkpoint utility
 *
 * Returns:
 *  1 if the flag is supported,
 *  0 if the flag is not supported,
 *  -1 if there was an error launching the cuda-checkpoint utility.
 */
static int cuda_checkpoint_supports_flag(const char *flag)
{
	char msg_buf[2048];
	const char *args[] = { CUDA_CHECKPOINT, "-h", NULL };

	if (launch_cuda_checkpoint(args, msg_buf, sizeof(msg_buf)) != 0)
		return -1;

	if (strstr(msg_buf, flag) == NULL)
		return 0;

	return 1;
}

/* Retrieve the cuda restore thread TID from the root pid */
static int get_cuda_restore_tid(int root_pid)
{
	char pid_buf[16];
	char pid_out[CUDA_CKPT_BUF_SIZE];

	snprintf(pid_buf, sizeof(pid_buf), "%d", root_pid);

	const char *args[] = { CUDA_CHECKPOINT, "--get-restore-tid", "--pid", pid_buf, NULL };
	int ret = launch_cuda_checkpoint(args, pid_out, sizeof(pid_out));
	if (ret != 0) {
		pr_err("Failed to launch cuda-checkpoint to retrieve restore tid: %s\n", pid_out);
		return -1;
	}

	return atoi(pid_out);
}

static cuda_task_state_t get_task_state_enum(const char *state_str)
{
	if (strncmp(state_str, "running", 7) == 0)
		return CUDA_TASK_RUNNING;

	if (strncmp(state_str, "locked", 6) == 0)
		return CUDA_TASK_LOCKED;

	if (strncmp(state_str, "checkpointed", 12) == 0)
		return CUDA_TASK_CHECKPOINTED;

	pr_err("Unknown CUDA state: %s\n", state_str);
	return CUDA_TASK_UNKNOWN;
}

static cuda_task_state_t get_cuda_state(pid_t pid)
{
	char pid_buf[16];
	char state_str[CUDA_CKPT_BUF_SIZE];
	const char *args[] = { CUDA_CHECKPOINT, "--get-state", "--pid", pid_buf, NULL };

	snprintf(pid_buf, sizeof(pid_buf), "%d", pid);

	if (launch_cuda_checkpoint(args, state_str, sizeof(state_str))) {
		pr_err("Failed to launch cuda-checkpoint to retrieve state: %s\n", state_str);
		return CUDA_TASK_UNKNOWN;
	}

	return get_task_state_enum(state_str);
}

static int cuda_process_checkpoint_action(int pid, const char *action, unsigned int timeout, char *msg_buf,
					  int buf_size)
{
	char pid_buf[16];
	char timeout_buf[16];

	snprintf(pid_buf, sizeof(pid_buf), "%d", pid);

	const char *args[] = { CUDA_CHECKPOINT, "--action", action, "--pid", pid_buf, NULL /* --timeout */,
			       NULL /* timeout_val */, NULL };
	if (timeout > 0) {
		snprintf(timeout_buf, sizeof(timeout_buf), "%d", timeout);
		args[5] = "--timeout";
		args[6] = timeout_buf;
	}

	return launch_cuda_checkpoint(args, msg_buf, buf_size);
}

static int interrupt_restore_thread(int restore_tid, k_rtsigset_t *restore_sigset)
{
	/* Since we resumed a thread that CRIU previously already froze we need to
	 * INTERRUPT it once again, task was already SEIZE'd so we don't need to do
	 * a compel_interrupt_task()
	 */
	if (ptrace(PTRACE_INTERRUPT, restore_tid, NULL, 0)) {
		pr_perror("Could not interrupt cuda restore tid %d after checkpoint, process may be in strange state",
			  restore_tid);
		return -1;
	}

	struct proc_status_creds creds;
	if (compel_wait_task(restore_tid, -1, parse_pid_status, NULL, &creds.s, NULL) != COMPEL_TASK_ALIVE) {
		pr_err("compel_wait_task failed after interrupt\n");
		return -1;
	}

	if (ptrace(PTRACE_SETOPTIONS, restore_tid, NULL, PTRACE_O_SUSPEND_SECCOMP | PTRACE_O_TRACESYSGOOD)) {
		pr_perror("Failed to set ptrace options on interrupt for restore tid %d", restore_tid);
		return -1;
	}

	if (ptrace(PTRACE_SETSIGMASK, restore_tid, sizeof(*restore_sigset), restore_sigset)) {
		pr_perror("Unable to restore original sigmask to restore tid %d", restore_tid);
		return -1;
	}

	return 0;
}

static int resume_restore_thread(int restore_tid, k_rtsigset_t *save_sigset)
{
	k_rtsigset_t block;

	if (ptrace(PTRACE_GETSIGMASK, restore_tid, sizeof(*save_sigset), save_sigset)) {
		pr_perror("Failed to get current sigmask for restore tid %d", restore_tid);
		return -1;
	}

	ksigfillset(&block);
	ksigdelset(&block, SIGTRAP);

	if (ptrace(PTRACE_SETSIGMASK, restore_tid, sizeof(block), &block)) {
		pr_perror("Failed to block signals on restore tid %d", restore_tid);
		return -1;
	}

	// Clear out PTRACE_O_SUSPEND_SECCOMP when we resume the restore thread
	if (ptrace(PTRACE_SETOPTIONS, restore_tid, NULL, 0)) {
		pr_perror("Could not clear ptrace options on restore tid %d", restore_tid);
		return -1;
	}

	if (ptrace(PTRACE_CONT, restore_tid, NULL, 0)) {
		pr_perror("Could not resume cuda restore tid %d", restore_tid);
		return -1;
	}

	return 0;
}

int cuda_plugin_checkpoint_devices(int pid)
{
	int restore_tid;
	char msg_buf[CUDA_CKPT_BUF_SIZE];
	int int_ret;
	int status;
	k_rtsigset_t save_sigset;
	struct pid_info *task_info;
	bool pid_found = false;
	struct gpu_region *vmas_before = NULL, *vmas_after = NULL, *new_vmas = NULL;
	int n_before = 0, n_after = 0, n_new = 0;

	if (plugin_disabled) {
		return -ENOTSUP;
	}

	restore_tid = get_cuda_restore_tid(pid);

	/* We can possibly hit a race with cuInit() where we are past the point of
	 * locking the process but at lock time cuInit() hadn't completed in which
	 * case cuda-checkpoint will report that we're in an invalid state to
	 * checkpoint
	 */
	if (restore_tid == -1) {
		pr_info("No need to checkpoint devices on pid %d\n", pid);
		return 0;
	}

	/* Check if the process is already in a checkpointed state */
	list_for_each_entry(task_info, &cuda_pids, list) {
		if (task_info->pid == pid) {
			if (task_info->initial_task_state == CUDA_TASK_CHECKPOINTED) {
				pr_info("pid %d already in a checkpointed state\n", pid);
				return 0;
			}
			pid_found = true;
			break;
		}
	}

	if (pid_found == false) {
		/* We return an error here. The task should be restored
		 * to its original state at cuda_plugin_fini().
		 */
		pr_err("Failed to track pid %d\n", pid);
		return -1;
	}

	/* Pre-scan: record anonymous private VMAs before cuda-checkpoint moves VRAM */
	if (scan_anon_private_vmas(pid, &vmas_before, &n_before) != 0)
		pr_warn("Pre-scan failed for pid %d, fast GPU page dump disabled\n", pid);

	pr_info("Checkpointing CUDA devices on pid %d restore_tid %d\n", pid, restore_tid);
	/* We need to resume the checkpoint thread to prepare the mappings for
	 * checkpointing
	 */
	if (resume_restore_thread(restore_tid, &save_sigset)) {
		free(vmas_before);
		return -1;
	}

	task_info->checkpointed = 1;
	{
		double t0 = now_ms();
		status = cuda_process_checkpoint_action(pid, ACTION_CHECKPOINT, 0, msg_buf, sizeof(msg_buf));
		pr_info("[timing] cuda-checkpoint checkpoint: %.0f ms\n", now_ms() - t0);
	}
	if (status) {
		pr_err("CHECKPOINT_DEVICES failed with %s\n", msg_buf);
	}

	int_ret = interrupt_restore_thread(restore_tid, &save_sigset);

	/* Fast GPU page dump: find new VMAs created by cuda-checkpoint, dump them
	 * with process_vm_readv, then free with injected madvise(DONTNEED) so
	 * CRIU's page walk sees empty pages and skips them.
	 */
	if (status == 0 && int_ret == 0 && vmas_before != NULL) {
		double t0, total_bytes = 0;
		int img_dir_fd = criu_get_image_dir();
		uint64_t syscall_addr;
		int i;

		if (img_dir_fd < 0) {
			pr_warn("No image dir fd, skipping fast GPU page dump\n");
			goto done;
		}

		syscall_addr = find_syscall_addr(pid);
		if (!syscall_addr) {
			pr_warn("Could not find syscall insn in vdso for pid %d, skipping fast GPU page dump\n", pid);
			goto done;
		}
		pr_info("Found syscall insn at 0x%llx for pid %d\n",
			(unsigned long long)syscall_addr, pid);

		t0 = now_ms();
		if (scan_anon_private_vmas(pid, &vmas_after, &n_after) != 0) {
			pr_warn("Post-scan failed for pid %d, skipping fast GPU page dump\n", pid);
			goto done;
		}
		if (diff_anon_vmas(vmas_before, n_before, vmas_after, n_after, &new_vmas, &n_new) != 0) {
			pr_warn("VMA diff failed for pid %d, skipping fast GPU page dump\n", pid);
			goto done;
		}
		pr_info("[timing] post-scan+diff: %.0f ms, %d new VMAs\n", now_ms() - t0, n_new);
		if (n_new == 0) {
			pr_info("No new GPU VMAs found for pid %d\n", pid);
			goto done;
		}
		for (i = 0; i < n_new; i++)
			total_bytes += new_vmas[i].size;
		pr_info("Found %d new GPU VMAs for pid %d (%.0f MB), dumping with process_vm_readv\n",
			n_new, pid, total_bytes / (1024 * 1024));

		t0 = now_ms();
		if (dump_gpu_pages(pid, img_dir_fd, new_vmas, n_new) == 0) {
			double dump_ms = now_ms() - t0;

			pr_info("[timing] process_vm_readv dump: %.0f ms (%.1f GB/s)\n",
				dump_ms, total_bytes / dump_ms / 1e6);
			t0 = now_ms();
			if (release_gpu_pages(pid, syscall_addr, new_vmas, n_new) != 0)
				pr_warn("madvise(DONTNEED) injection failed for pid %d, CRIU will dump GPU pages slowly\n",
					pid);
			else
				pr_info("[timing] injected madvise(DONTNEED): %.0f ms\n", now_ms() - t0);
		} else {
			pr_warn("Fast GPU page dump failed for pid %d, CRIU will dump GPU pages\n", pid);
		}
	}

done:
	free(vmas_before);
	free(vmas_after);
	free(new_vmas);
	return status != 0 ? -1 : int_ret;
}
CR_PLUGIN_REGISTER_HOOK(CR_PLUGIN_HOOK__CHECKPOINT_DEVICES, cuda_plugin_checkpoint_devices);

int cuda_plugin_pause_devices(int pid)
{
	int restore_tid;
	char msg_buf[CUDA_CKPT_BUF_SIZE];
	cuda_task_state_t task_state;

	if (plugin_disabled) {
		return -ENOTSUP;
	}

	restore_tid = get_cuda_restore_tid(pid);

	if (restore_tid == -1) {
		pr_info("no need to pause devices on pid %d\n", pid);
		return 0;
	}

	task_state = get_cuda_state(restore_tid);
	if (task_state == CUDA_TASK_UNKNOWN) {
		pr_err("Failed to get CUDA state for PID %d\n", restore_tid);
		return -1;
	}

	if (!plugin_added_to_inventory) {
		if (add_inventory_plugin(CR_PLUGIN_DESC.name)) {
			pr_err("Failed to add CUDA plugin to inventory image\n");
			return -1;
		}
		plugin_added_to_inventory = true;
	}

	if (task_state == CUDA_TASK_LOCKED) {
		pr_info("pid %d already in a locked state\n", pid);
		/* Leave this PID in a "locked" state at resume_device() */
		add_pid_to_buf(&cuda_pids, pid, CUDA_TASK_LOCKED);
		return 0;
	}

	if (task_state == CUDA_TASK_CHECKPOINTED) {
		/* We need to skip this PID in cuda_plugin_checkpoint_devices(),
		 * and leave it in a "checkpoined" state at resume_device(). */
		add_pid_to_buf(&cuda_pids, pid, CUDA_TASK_CHECKPOINTED);
		return 0;
	}

	pr_info("pausing devices on pid %d\n", pid);
	int status = cuda_process_checkpoint_action(pid, ACTION_LOCK, opts.timeout * 1000, msg_buf, sizeof(msg_buf));
	if (status) {
		pr_err("PAUSE_DEVICES failed with %s\n", msg_buf);
		if (alarm_timeouted())
			goto unlock;
		return -1;
	}

	if (add_pid_to_buf(&cuda_pids, pid, CUDA_TASK_RUNNING)) {
		pr_err("unable to track paused pid %d\n", pid);
		goto unlock;
	}

	return 0;
unlock:
	status = cuda_process_checkpoint_action(pid, ACTION_UNLOCK, 0, msg_buf, sizeof(msg_buf));
	if (status) {
		pr_err("Failed to unlock process status %s, pid %d may hang\n", msg_buf, pid);
	}
	return -1;
}
CR_PLUGIN_REGISTER_HOOK(CR_PLUGIN_HOOK__PAUSE_DEVICES, cuda_plugin_pause_devices)

int resume_device(int pid, int checkpointed, cuda_task_state_t initial_task_state)
{
	char msg_buf[CUDA_CKPT_BUF_SIZE];
	int status;
	int ret = 0;
	int int_ret;
	k_rtsigset_t save_sigset;

	if (initial_task_state == CUDA_TASK_UNKNOWN) {
		pr_info("skip resume for PID %d (unknown state)\n", pid);
		return 0;
	}

	int restore_tid = get_cuda_restore_tid(pid);
	if (restore_tid == -1) {
		pr_info("No need to resume devices on pid %d\n", pid);
		return 0;
	}

	pr_info("resuming devices on pid %d\n", pid);
	/* The resuming process has to stay frozen during this time otherwise
	 * attempting to access a UVM pointer will crash if we haven't restored the
	 * underlying mappings yet
	 */
	pr_debug("Restore thread pid %d found for real pid %d\n", restore_tid, pid);
	/* wakeup the restore thread so we can handle the restore for this pid,
	 * rseq_cs has to be restored before execution
	 */
	if (resume_restore_thread(restore_tid, &save_sigset)) {
		return -1;
	}

	if (checkpointed && (initial_task_state == CUDA_TASK_RUNNING || initial_task_state == CUDA_TASK_LOCKED)) {
		/* If the process was "locked" or "running" before checkpointing it, we need to restore it */
		status = cuda_process_checkpoint_action(pid, ACTION_RESTORE, 0, msg_buf, sizeof(msg_buf));
		if (status) {
			pr_err("RESUME_DEVICES RESTORE failed with %s\n", msg_buf);
			ret = -1;
			goto interrupt;
		}
	}

	if (initial_task_state == CUDA_TASK_RUNNING) {
		/* If the process was "running" before we paused it, we need to unlock it */
		status = cuda_process_checkpoint_action(pid, ACTION_UNLOCK, 0, msg_buf, sizeof(msg_buf));
		if (status) {
			pr_err("RESUME_DEVICES UNLOCK failed with %s\n", msg_buf);
			ret = -1;
		}
	}

interrupt:
	int_ret = interrupt_restore_thread(restore_tid, &save_sigset);

	return ret != 0 ? ret : int_ret;
}

int cuda_plugin_resume_devices_late(int pid)
{
	int img_dir_fd;
	int restore_tid;
	uint64_t syscall_addr;

	if (plugin_disabled) {
		return -ENOTSUP;
	}

	img_dir_fd = criu_get_image_dir();
	restore_tid = get_cuda_restore_tid(pid);

	/* If CUDA_PLUGIN_SKIP_RESTORE is set to 1/true/yes, the orchestrator will
	 * handle GPU restore externally (after moving the process into the GPU
	 * cgroup). The dump was produced by cuda-offload which writes a
	 * cuda-offload-restore marker alongside the image. Skip both
	 * restore_gpu_pages and resume_device. Without the env var, fall through
	 * to normal restore so plain `criu restore` still works out of the box.
	 */
	{
		const char *skip = getenv("CUDA_PLUGIN_SKIP_RESTORE");
		if (skip) {
			char tmp[8];
			strncpy(tmp, skip, sizeof(tmp) - 1);
			tmp[sizeof(tmp) - 1] = '\0';
			for (int j = 0; tmp[j]; j++)
				tmp[j] = tolower((unsigned char)tmp[j]);
			if (strcmp(tmp, "1") == 0 || strcmp(tmp, "true") == 0 || strcmp(tmp, "yes") == 0) {
				pr_info("CUDA_PLUGIN_SKIP_RESTORE set, deferring restore to cuda-offload for pid %d\n", pid);
				return 0;
			}
		}
	}

	if (img_dir_fd >= 0 && restore_tid != -1) {
		syscall_addr = find_syscall_addr(pid);
		if (!syscall_addr) {
			pr_warn("Could not find syscall insn in vdso for pid %d\n", pid);
		} else {
			double t0 = now_ms();

			if (restore_gpu_pages(pid, pid, syscall_addr, img_dir_fd) != 0)
				pr_warn("O_DIRECT pread restore failed for pid %d\n", pid);
			else
				pr_info("[timing] O_DIRECT pread restore: %.0f ms\n", now_ms() - t0);
		}
	} else if (img_dir_fd < 0) {
		pr_warn("No image dir fd during restore for pid %d\n", pid);
	}

	/* RESUME_DEVICES_LATE is used during `criu restore`.
	 * Here, we assume that users expect the target process
	 * to be in a "running" state after restore, even if it was
	 * in a "locked" or "checkpointed" state during `criu dump`.
	 */
	{
		double t0 = now_ms();
		int r = resume_device(pid, 1, CUDA_TASK_RUNNING);

		pr_info("[timing] cuda-checkpoint restore+unlock: %.0f ms\n", now_ms() - t0);
		return r;
	}
}
CR_PLUGIN_REGISTER_HOOK(CR_PLUGIN_HOOK__RESUME_DEVICES_LATE, cuda_plugin_resume_devices_late)

/**
 * Check if a CUDA device is available on the system
 */
static bool is_cuda_device_available(void)
{
	const char *gpu_path = "/proc/driver/nvidia/gpus/";
	struct stat sb;

	if (stat(gpu_path, &sb) != 0)
		return false;

	return S_ISDIR(sb.st_mode);
}

int cuda_plugin_init(int stage)
{
	int ret;

	/* Disable CUDA checkpointing with pre-dump */
	if (stage == CR_PLUGIN_STAGE__PRE_DUMP) {
		plugin_disabled = true;
		return 0;
	}

	if (stage == CR_PLUGIN_STAGE__RESTORE) {
		if (!check_and_remove_inventory_plugin(CR_PLUGIN_DESC.name, strlen(CR_PLUGIN_DESC.name))) {
			plugin_disabled = true;
			return 0;
		}
	}

	if (!fault_injected(FI_PLUGIN_CUDA_FORCE_ENABLE) && !is_cuda_device_available()) {
		pr_info("No GPU device found; CUDA plugin is disabled\n");
		plugin_disabled = true;
		return 0;
	}

	ret = cuda_checkpoint_supports_flag("--action");
	if (ret == -1) {
		pr_warn("check that %s is present in $PATH\n", CUDA_CHECKPOINT);
		plugin_disabled = true;
		return 0;
	}

	if (ret == 0) {
		pr_warn("cuda-checkpoint --action flag not supported, an r555 or higher version driver is required. Disabling CUDA plugin\n");
		plugin_disabled = true;
		return 0;
	}

	pr_info("initialized: %s stage %d\n", CR_PLUGIN_DESC.name, stage);

	/* In the DUMP stage track all the PID's we've paused CUDA operations on to
	 * release them when we're done if the user requested the leave-running option
	 */
	if (stage == CR_PLUGIN_STAGE__DUMP) {
		INIT_LIST_HEAD(&cuda_pids);
	}

	set_compel_interrupt_only_mode();

	return 0;
}

void cuda_plugin_fini(int stage, int ret)
{
	if (plugin_disabled) {
		return;
	}

	pr_info("finished %s stage %d err %d\n", CR_PLUGIN_DESC.name, stage, ret);

	/* Release all the paused PID's at the end of the DUMP stage in case the
	 * user provides the -R (leave-running) flag or an error occurred
	 */
	if (stage == CR_PLUGIN_STAGE__DUMP && (opts.final_state == TASK_ALIVE || ret != 0)) {
		struct pid_info *info;
		list_for_each_entry(info, &cuda_pids, list) {
			resume_device(info->pid, info->checkpointed, info->initial_task_state);
		}
	}
	if (stage == CR_PLUGIN_STAGE__DUMP) {
		dealloc_pid_buffer(&cuda_pids);
	}
}
CR_PLUGIN_REGISTER("cuda_plugin", cuda_plugin_init, cuda_plugin_fini)
