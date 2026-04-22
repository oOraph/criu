/*
 * cuda_gpu_pages.c - shared GPU VRAM page I/O routines
 *
 * See cuda_gpu_pages.h for the interface description.
 */

#include "cuda_gpu_pages.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#ifndef SYS_open_tree
#define SYS_open_tree 428
#endif

#ifndef SYS_move_mount
#define SYS_move_mount 429
#endif

#ifndef SYS_setns
#define SYS_setns 308
#endif

#ifndef SYS_umount2
#define SYS_umount2 166
#endif

#ifndef OPEN_TREE_CLONE
#define OPEN_TREE_CLONE 1
#endif

#ifndef MOVE_MOUNT_F_EMPTY_PATH
#define MOVE_MOUNT_F_EMPTY_PATH 0x00000004
#endif

#ifndef CLONE_NEWNS
#define CLONE_NEWNS 0x00020000
#endif

#ifndef AT_NO_AUTOMOUNT
#define AT_NO_AUTOMOUNT 0x800
#endif

#ifndef MNT_DETACH
#define MNT_DETACH 2
#endif

#ifndef SYS_process_vm_readv
#define SYS_process_vm_readv 310
#endif

#ifndef SYS_process_vm_writev
#define SYS_process_vm_writev 311
#endif

#ifndef SYS_mlock
#define SYS_mlock 149
#endif

#ifndef SYS_munlock
#define SYS_munlock 150
#endif

#ifndef SYS_mmap
#define SYS_mmap 9
#endif

#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif

#ifndef SYS_pread64
#define SYS_pread64 17
#endif

#ifndef O_DIRECT
#define O_DIRECT 040000 /* Linux x86-64 */
#endif

#ifndef O_PATH
#define O_PATH 010000000 /* Linux x86-64 */
#endif

#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 14
#endif

#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif

#ifdef BUILD_FOR_CRIU_PLUGIN
#include "log.h" /* routes pr_* to CRIU's log fd (restore.log) */
#else
#define pr_info(fmt, ...)   fprintf(stderr, "cuda_gpu_pages: " fmt, ##__VA_ARGS__)
#define pr_warn(fmt, ...)   fprintf(stderr, "cuda_gpu_pages: WARNING: " fmt, ##__VA_ARGS__)
#define pr_err(fmt, ...)    fprintf(stderr, "cuda_gpu_pages: ERROR: " fmt, ##__VA_ARGS__)
#define pr_perror(fmt, ...) fprintf(stderr, "cuda_gpu_pages: ERROR: " fmt ": %s\n", \
				    ##__VA_ARGS__, strerror(errno))
#endif

double now_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/*
 * Return the innermost namespace PID for 'pid' by reading NSpid from
 * /proc/<pid>/status.  The host PID changes on every restore cycle (CRIU
 * preserves the in-container PID, not the host PID), so we key the
 * gpu-pages image file on the namespace PID to get a stable filename.
 * Falls back to 'pid' if the info is unavailable.
 *
 * Limitation: this only works when CRIU preserves namespace PIDs, which
 * is the normal case for same-node scale-to-zero (the PID namespace is
 * recreated fresh each restore).  If namespace PIDs also change (e.g.
 * live migration to another node where the namespace cannot be restored
 * cleanly), this will still produce ENOENT.  The proper fix for that case
 * is a BFS positional pid mapping, the same approach used in cuda-offload.c
 * (gpu-offload-pids.img): save the BFS-ordered checkpoint pid list at dump
 * time, then at restore time walk the live tree in the same BFS order and
 * map ckpt_pids[i] -> live_pids[i] to find the right image file.
 */
static int get_ns_pid(int pid)
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
 * Scan /proc/<pid>/maps for anonymous private rw- VMAs.
 * Anonymous = dev 0:0, ino 0, no filename.
 */
int scan_anon_private_vmas(int pid, struct gpu_region **out, int *count)
{
	char maps_path[64];
	FILE *f;
	char line[256];
	struct gpu_region *regions = NULL;
	int n = 0, cap = 0;

	snprintf(maps_path, sizeof(maps_path), "/proc/%d/maps", pid);
	f = fopen(maps_path, "r");
	if (!f) {
		pr_perror("Cannot open %s", maps_path);
		return -1;
	}

	while (fgets(line, sizeof(line), f)) {
		unsigned long start, end, offset, ino;
		unsigned int dev_maj, dev_min;
		char perms[8];
		char name[128];
		int n_parsed;
		struct gpu_region *tmp;
		int new_cap;

		name[0] = '\0';
		n_parsed = sscanf(line, "%lx-%lx %7s %lx %x:%x %lu %127s",
				  &start, &end, perms, &offset,
				  &dev_maj, &dev_min, &ino, name);
		if (n_parsed < 7)
			continue;

		/* anonymous: dev=0:0, ino=0, no filename */
		if (dev_maj != 0 || dev_min != 0 || ino != 0)
			continue;
		if (n_parsed >= 8 && name[0] != '\0')
			continue;

		/* private, read-write */
		if (perms[3] != 'p' || perms[0] != 'r' || perms[1] != 'w')
			continue;

		if (n >= cap) {
			new_cap = cap ? cap * 2 : 64;
			tmp = realloc(regions, (size_t)new_cap * sizeof(*regions));
			if (!tmp) {
				pr_err("OOM in scan_anon_private_vmas\n");
				fclose(f);
				free(regions);
				return -1;
			}
			regions = tmp;
			cap = new_cap;
		}

		regions[n].start = (uint64_t)start;
		regions[n].size = (uint64_t)(end - start);
		n++;
	}

	fclose(f);
	*out = regions;
	*count = n;
	return 0;
}

/*
 * Find VMAs present in 'after' but not in 'before'.
 * These are the new anonymous mappings created by cuda-checkpoint for VRAM.
 */
int diff_anon_vmas(struct gpu_region *before, int n_before,
		   struct gpu_region *after, int n_after,
		   struct gpu_region **diff_out, int *diff_count)
{
	struct gpu_region *diff = NULL;
	int n = 0, cap = 0;
	int i, j;

	for (i = 0; i < n_after; i++) {
		bool found = false;

		for (j = 0; j < n_before; j++) {
			if (before[j].start == after[i].start && before[j].size == after[i].size) {
				found = true;
				break;
			}
		}
		if (!found) {
			struct gpu_region *tmp;
			int new_cap;

			if (n >= cap) {
				new_cap = cap ? cap * 2 : 16;
				tmp = realloc(diff, (size_t)new_cap * sizeof(*diff));
				if (!tmp) {
					pr_err("OOM in diff_anon_vmas\n");
					free(diff);
					return -1;
				}
				diff = tmp;
				cap = new_cap;
			}
			diff[n++] = after[i];
		}
	}

	*diff_out = diff;
	*diff_count = n;
	return 0;
}

/*
 * Dump GPU memory regions to gpu-pages-<pid>.img in the image directory.
 * File format: gpu_pages_hdr | gpu_region[num_regions] | raw page data
 * Page data starts at GPU_PAGES_DATA_OFFSET (page-aligned for mmap on restore).
 */
int dump_gpu_pages(int pid, int img_dir_fd, struct gpu_region *regions, int count)
{
	char fname[64];
	int fd, ret = -1, i;
	struct gpu_pages_hdr hdr;
	char *buf = NULL;
	double t0, t_readv = 0, t_write = 0;

	int ns_pid = get_ns_pid(pid);

	hdr.magic = GPU_PAGES_MAGIC;
	hdr.num_regions = (uint32_t)count;

	pr_info("dump_gpu_pages: host_pid=%d ns_pid=%d\n", pid, ns_pid);
	snprintf(fname, sizeof(fname), "gpu-pages-%d.img", ns_pid);
	/*
	 * O_DIRECT bypasses the page cache: no dirty pages accumulate, so
	 * restore's O_DIRECT pread finds nothing to invalidate and runs at
	 * full NVMe bandwidth.  Without this, 6 GB of dirty pages require
	 * sync_file_range (blocks 4 s walking 1.5M page entries) before
	 * restore can read fast.  Header + regions are packed into one
	 * 4096-byte aligned block to meet O_DIRECT alignment requirements.
	 */
	fd = openat(img_dir_fd, fname,
		    O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0600);
	if (fd < 0 && errno == EINVAL) {
		pr_info("O_DIRECT not supported for dump, falling back to buffered I/O\n");
		fd = openat(img_dir_fd, fname,
			    O_WRONLY | O_CREAT | O_TRUNC, 0600);
	}
	if (fd < 0) {
		pr_perror("Cannot create %s", fname);
		return -1;
	}

	if (posix_memalign((void **)&buf, 4096, GPU_IO_CHUNK_SIZE) != 0) {
		pr_err("OOM: cannot allocate aligned IO buffer\n");
		goto out;
	}

	/* Pack header + region table into one 4096-byte O_DIRECT write */
	memset(buf, 0, GPU_PAGES_DATA_OFFSET);
	memcpy(buf, &hdr, sizeof(hdr));
	if (count > 0)
		memcpy(buf + sizeof(hdr), regions, (size_t)count * sizeof(*regions));
	if (write(fd, buf, GPU_PAGES_DATA_OFFSET) != (ssize_t)GPU_PAGES_DATA_OFFSET) {
		pr_perror("Cannot write header block to %s", fname);
		goto out;
	}

	t0 = now_ms();
	for (i = 0; i < count; i++) {
		uint64_t offset = 0;
		uint64_t remaining = regions[i].size;

		while (remaining > 0) {
			size_t chunk = (remaining > GPU_IO_CHUNK_SIZE) ? GPU_IO_CHUNK_SIZE : (size_t)remaining;
			struct iovec local_iov = { .iov_base = buf, .iov_len = chunk };
			struct iovec remote_iov = { .iov_base = (void *)(uintptr_t)(regions[i].start + offset),
						    .iov_len = chunk };
			ssize_t n, written = 0;
			double t1;

			t1 = now_ms();
			n = (ssize_t)syscall(SYS_process_vm_readv, (pid_t)pid, &local_iov, 1UL, &remote_iov, 1UL, 0UL);
			t_readv += now_ms() - t1;
			if (n < 0) {
				pr_perror("process_vm_readv failed for pid %d at 0x%lx", pid,
					  (unsigned long)(regions[i].start + offset));
				goto out;
			}

			t1 = now_ms();
			while (written < n) {
				ssize_t w = write(fd, buf + written, (size_t)(n - written));

				if (w < 0) {
					pr_perror("write to %s failed", fname);
					goto out;
				}
				written += w;
			}
			t_write += now_ms() - t1;
			offset += (uint64_t)n;
			remaining -= (uint64_t)n;
		}
	}

	pr_info("[timing] dump: process_vm_readv=%.0f ms O_DIRECT_write=%.0f ms total=%.0f ms\n",
		t_readv, t_write, now_ms() - t0);
	ret = 0;
	pr_info("Dumped %d GPU regions for pid %d\n", count, pid);
out:
	free(buf);
	close(fd);
	if (ret != 0)
		unlinkat(img_dir_fd, fname, 0);
	return ret;
}

/*
 * Find a 'syscall' instruction (0x0f 0x05) in the vdso of pid.
 * Returns the address, or 0 on failure.
 */
uint64_t find_syscall_addr(int pid)
{
	char maps_path[64];
	FILE *f;
	char line[256];
	uint64_t vdso_start = 0, vdso_end = 0;
	uint8_t *buf;
	size_t vdso_size, i;
	struct iovec local_iov, remote_iov;

	snprintf(maps_path, sizeof(maps_path), "/proc/%d/maps", pid);
	f = fopen(maps_path, "r");
	if (!f)
		return 0;

	while (fgets(line, sizeof(line), f)) {
		unsigned long start, end;
		char name[64];

		name[0] = '\0';
		if (sscanf(line, "%lx-%lx %*s %*s %*s %*s %63s", &start, &end, name) >= 2 &&
		    strcmp(name, "[vdso]") == 0) {
			vdso_start = start;
			vdso_end = end;
			break;
		}
	}
	fclose(f);

	if (!vdso_start)
		return 0;

	vdso_size = vdso_end - vdso_start;
	buf = malloc(vdso_size);
	if (!buf)
		return 0;

	local_iov.iov_base = buf;
	local_iov.iov_len = vdso_size;
	remote_iov.iov_base = (void *)(uintptr_t)vdso_start;
	remote_iov.iov_len = vdso_size;

	if (syscall(SYS_process_vm_readv, (pid_t)pid, &local_iov, 1UL, &remote_iov, 1UL, 0UL) < 0) {
		free(buf);
		return 0;
	}

	for (i = 0; i + 1 < vdso_size; i++) {
		if (buf[i] == 0x0f && buf[i + 1] == 0x05) { /* syscall */
			free(buf);
			return vdso_start + i;
		}
	}

	free(buf);
	return 0;
}

/*
 * Generic 6-argument syscall injector for a stopped thread.
 * syscall_addr must point to a 'syscall' (0x0f 0x05) instruction in the
 * target's vdso. The thread must be in ptrace-stop state.
 * Returns the syscall return value (rax), or LONG_MIN on ptrace error.
 */
long inject_syscall(int tid, uint64_t syscall_addr,
		    long nr, long a1, long a2, long a3,
		    long a4, long a5, long a6)
{
	struct user_regs_struct saved_regs, regs, after_regs;
	int status;

	if (ptrace(PTRACE_GETREGS, tid, NULL, &saved_regs) < 0) {
		pr_perror("PTRACE_GETREGS failed for tid %d", tid);
		return LONG_MIN;
	}

	regs = saved_regs;
	regs.rax = (unsigned long long)nr;
	regs.rdi = (unsigned long long)a1;
	regs.rsi = (unsigned long long)a2;
	regs.rdx = (unsigned long long)a3;
	regs.r10 = (unsigned long long)a4;
	regs.r8  = (unsigned long long)a5;
	regs.r9  = (unsigned long long)a6;
	regs.rip = syscall_addr;
	regs.orig_rax = (unsigned long long)-1; /* not in a syscall-stop */

	if (ptrace(PTRACE_SETREGS, tid, NULL, &regs) < 0) {
		pr_perror("PTRACE_SETREGS failed for tid %d", tid);
		ptrace(PTRACE_SETREGS, tid, NULL, &saved_regs);
		return LONG_MIN;
	}

	if (ptrace(PTRACE_SINGLESTEP, tid, NULL, NULL) < 0) {
		pr_perror("PTRACE_SINGLESTEP failed for tid %d", tid);
		ptrace(PTRACE_SETREGS, tid, NULL, &saved_regs);
		return LONG_MIN;
	}

	if (waitpid(tid, &status, __WALL) < 0) {
		pr_perror("waitpid after SINGLESTEP failed for tid %d", tid);
		ptrace(PTRACE_SETREGS, tid, NULL, &saved_regs);
		return LONG_MIN;
	}

	if (ptrace(PTRACE_GETREGS, tid, NULL, &after_regs) < 0) {
		pr_perror("PTRACE_GETREGS after syscall failed for tid %d", tid);
		ptrace(PTRACE_SETREGS, tid, NULL, &saved_regs);
		return LONG_MIN;
	}

	if (ptrace(PTRACE_SETREGS, tid, NULL, &saved_regs) < 0) {
		pr_perror("PTRACE_SETREGS restore failed for tid %d", tid);
		return LONG_MIN;
	}

	return (long)after_regs.rax;
}

/*
 * Inject madvise(addr, len, MADV_DONTNEED) into the stopped thread 'tid'.
 */
int inject_madvise_dontneed(int tid, uint64_t addr, uint64_t len, uint64_t syscall_addr)
{
	long ret = inject_syscall(tid, syscall_addr,
				  SYS_madvise, (long)addr, (long)len,
				  MADV_DONTNEED, 0, 0, 0);

	if (ret < 0) {
		pr_warn("injected madvise(DONTNEED) returned %ld for addr=0x%lx len=%lu\n",
			ret, (unsigned long)addr, (unsigned long)len);
		return -1;
	}
	return 0;
}

/*
 * Free GPU pages from the stopped thread 'tid' by injecting madvise(MADV_DONTNEED).
 * The VMAs remain in the address space (empty), ready to be remapped on restore.
 * tid must be in ptrace-stop state; syscall_addr must be a 'syscall' insn in vdso.
 */
int release_gpu_pages(int tid, uint64_t syscall_addr, struct gpu_region *regions, int count)
{
	int i, failed = 0;

	for (i = 0; i < count; i++) {
		if (inject_madvise_dontneed(tid, regions[i].start, regions[i].size, syscall_addr) != 0) {
			pr_warn("inject madvise(DONTNEED) failed for region %d (0x%llx+%llu)\n",
				i, (unsigned long long)regions[i].start,
				(unsigned long long)regions[i].size);
			failed++;
		}
	}

	if (failed == 0)
		pr_info("Released %d GPU regions via injected madvise(DONTNEED)\n", count);
	else
		pr_warn("Released %d/%d GPU regions via injected madvise(DONTNEED)\n",
			count - failed, count);

	return (failed == count) ? -1 : 0;
}

/*
 * Enter pid's mount namespace, bind mount src (host absolute path) to dst
 * (a path inside the container, e.g. /tmp/.criu-gpu-restore.img), then
 * return to the original mount namespace.
 *
 * The destination file is created via /proc/<pid>/root<dst> before the
 * setns so the bind mount has a target to attach to.
 *
 * This works as long as the host filesystem path 'src' is accessible from
 * within the container's mount namespace.  For non-preserved-namespace
 * scenarios see the BFS fallback note in get_ns_pid().
 */
/*
 * Bind mount 'src' (host absolute path) to 'dst' inside the container's
 * mount namespace, using the same open_tree + move_mount pattern that CRIU
 * uses in do_mount_in_right_mntns() (criu/mount-v2.c).
 *
 * open_tree(OPEN_TREE_CLONE) captures a detached anonymous mount of 'src'
 * while still in the host namespace — the source path does not need to be
 * accessible from inside the container.  After setns into the container's
 * mount namespace, move_mount attaches the detached mount at 'dst'.
 */
static int bind_mount_in_container(int pid, const char *src, const char *dst)
{
	char proc_dst[PATH_MAX];
	char ns_path[64];
	int tree_fd = -1, saved_mns_fd = -1, container_mns_fd = -1, cwd_fd = -1;
	int fd, ret = -1;

	/* Capture a detached clone of src's mount in the host namespace. */
	tree_fd = (int)syscall(SYS_open_tree, AT_FDCWD, src,
			       OPEN_TREE_CLONE | AT_NO_AUTOMOUNT | AT_SYMLINK_NOFOLLOW);
	if (tree_fd < 0) {
		pr_perror("open_tree %s failed", src);
		return -1;
	}

	/* Create the destination file inside the container's rootfs. */
	snprintf(proc_dst, sizeof(proc_dst), "/proc/%d/root%s", pid, dst);
	fd = open(proc_dst, O_CREAT | O_WRONLY | O_TRUNC, 0600);
	if (fd < 0) {
		pr_perror("Cannot create bind mount destination %s", proc_dst);
		goto out;
	}
	close(fd);

	saved_mns_fd = open("/proc/self/ns/mnt", O_RDONLY | O_CLOEXEC);
	if (saved_mns_fd < 0) {
		pr_perror("Cannot open /proc/self/ns/mnt");
		goto out;
	}

	/*
	 * Save CWD before entering the container namespace.  The container's
	 * mnt ns may not contain the host work directory, leaving the CWD
	 * detached after setns; fchdir restores it on the way back so that
	 * CRIU's relative paths (e.g. .criu.cgyard.*) remain resolvable.
	 */
	cwd_fd = open(".", O_PATH | O_DIRECTORY);
	if (cwd_fd < 0) {
		pr_perror("Cannot open current directory");
		goto out;
	}

	snprintf(ns_path, sizeof(ns_path), "/proc/%d/ns/mnt", pid);
	container_mns_fd = open(ns_path, O_RDONLY | O_CLOEXEC);
	if (container_mns_fd < 0) {
		pr_perror("Cannot open container mount namespace %s", ns_path);
		goto out;
	}

	if (syscall(SYS_setns, container_mns_fd, CLONE_NEWNS) < 0) {
		pr_perror("setns to container mount namespace failed");
		goto out;
	}

	if (syscall(SYS_move_mount, tree_fd, "", AT_FDCWD, dst,
		    MOVE_MOUNT_F_EMPTY_PATH) < 0)
		pr_perror("move_mount %s -> %s failed", src, dst);
	else
		ret = 0;

	if (syscall(SYS_setns, saved_mns_fd, CLONE_NEWNS) < 0) {
		pr_perror("setns back to host mount namespace failed");
		ret = -1;
	} else if (fchdir(cwd_fd) < 0) {
		pr_perror("fchdir to restore working directory failed");
		ret = -1;
	}
out:
	if (tree_fd >= 0)
		close(tree_fd);
	if (saved_mns_fd >= 0)
		close(saved_mns_fd);
	if (container_mns_fd >= 0)
		close(container_mns_fd);
	if (cwd_fd >= 0)
		close(cwd_fd);
	return ret;
}

static void umount_in_container(int pid, const char *dst)
{
	char ns_path[64];
	int saved_mns_fd, container_mns_fd, cwd_fd;

	saved_mns_fd = open("/proc/self/ns/mnt", O_RDONLY | O_CLOEXEC);
	if (saved_mns_fd < 0)
		return;

	cwd_fd = open(".", O_PATH | O_DIRECTORY);
	if (cwd_fd < 0) {
		close(saved_mns_fd);
		return;
	}

	snprintf(ns_path, sizeof(ns_path), "/proc/%d/ns/mnt", pid);
	container_mns_fd = open(ns_path, O_RDONLY | O_CLOEXEC);
	if (container_mns_fd < 0) {
		close(saved_mns_fd);
		close(cwd_fd);
		return;
	}

	if (syscall(SYS_setns, container_mns_fd, CLONE_NEWNS) == 0) {
		syscall(SYS_umount2, dst, MNT_DETACH);
		if (syscall(SYS_setns, saved_mns_fd, CLONE_NEWNS) < 0)
			pr_perror("setns back to host mount namespace failed in umount");
		else if (fchdir(cwd_fd) < 0)
			pr_perror("fchdir to restore working directory failed in umount");
	}
	close(container_mns_fd);
	close(saved_mns_fd);
	close(cwd_fd);
}

/*
 * Restore GPU pages into the target process via injected O_DIRECT pread64.
 *
 * Instead of mmap(MAP_SHARED)+MADV_POPULATE_READ (which reads through the
 * page cache at ~1.5 GB/s due to per-page kernel overhead), we inject
 * O_DIRECT pread64 calls directly into the target's existing anonymous VMAs.
 *
 * O_DIRECT programs the NVMe controller to DMA data straight into the
 * target's physical pages — no page cache, no intermediate copy, no per-page
 * fault overhead.  Expected throughput: ~3 GB/s (raw NVMe sequential read).
 *
 * For each GPU VMA we inject:
 *   openat(O_RDONLY|O_DIRECT)            — once, reused across regions
 *   pread64(fd, vma_addr, chunk, offset)  — N chunks per region
 *   mlock(vma_addr, size)                — pin pages for cuda-checkpoint DMA
 *   close(fd)                            — once at the end
 *
 * O_DIRECT alignment requirements (all guaranteed):
 *   buffer: VMA addresses are page-aligned (4096)
 *   count:  region sizes are page multiples; chunk = GPU_IO_CHUNK_SIZE (64 MB)
 *   offset: GPU_PAGES_DATA_OFFSET = 4096; all region offsets are page multiples
 *
 * If O_DIRECT is not supported (EINVAL), fall back to plain O_RDONLY so the
 * injection still works (at page-cache speed, same as the old mmap approach).
 */
int restore_gpu_pages(int pid, int tid, uint64_t syscall_addr, int img_dir_fd)
{
	char fname[64];
	char img_dir_path[PATH_MAX - 64];
	char file_path[PATH_MAX];
	char proc_link[64];
	int img_fd = -1, ret = -1;
	struct gpu_pages_hdr hdr;
	struct gpu_region *regions = NULL;
	uint64_t file_offset, total_bytes = 0;
	uint32_t i;
	long target_fd;
	uint64_t path_addr;
	double t0;
	int ns_pid = get_ns_pid(pid);

	pr_info("restore_gpu_pages: host_pid=%d ns_pid=%d\n", pid, ns_pid);
	snprintf(fname, sizeof(fname), "gpu-pages-%d.img", ns_pid);
	img_fd = openat(img_dir_fd, fname, O_RDONLY);
	if (img_fd < 0) {
		if (errno == ENOENT) {
			pr_info("No gpu-pages file for pid %d (ns_pid=%d), skipping\n", pid, ns_pid);
			return 0;
		}
		pr_perror("Cannot open %s", fname);
		return -1;
	}

	if (read(img_fd, &hdr, sizeof(hdr)) != (ssize_t)sizeof(hdr) ||
	    hdr.magic != GPU_PAGES_MAGIC) {
		pr_err("Bad header in %s\n", fname);
		goto out;
	}

	if (hdr.num_regions == 0) {
		ret = 0;
		goto out;
	}

	regions = malloc(hdr.num_regions * sizeof(*regions));
	if (!regions) {
		pr_err("OOM in restore_gpu_pages\n");
		goto out;
	}
	if (read(img_fd, regions, hdr.num_regions * sizeof(*regions)) !=
	    (ssize_t)(hdr.num_regions * sizeof(*regions))) {
		pr_perror("Cannot read region table");
		goto out;
	}
	close(img_fd);
	img_fd = -1;

	/*
	 * Resolve the host absolute path of the image file, then bind mount it
	 * into the container's mount namespace at a per-pid well-known path.
	 * The target process runs in the container's mount namespace and cannot
	 * see /var/lib/zeropod/... directly, so injecting openat with the host
	 * path yields ENOENT.  The bind mount makes the file visible at a path
	 * the target can open.  Use ns_pid in the name to avoid collisions when
	 * multiple pids in the same tree are restored concurrently.
	 */
	char gpu_restore_tmp_path[64];
	snprintf(gpu_restore_tmp_path, sizeof(gpu_restore_tmp_path),
		 "/tmp/.criu-gpu-restore-%d.img", ns_pid);

	snprintf(proc_link, sizeof(proc_link), "/proc/self/fd/%d", img_dir_fd);
	{
		ssize_t n = readlink(proc_link, img_dir_path, sizeof(img_dir_path) - 1);

		if (n < 0) {
			pr_perror("readlink %s failed", proc_link);
			goto out;
		}
		img_dir_path[n] = '\0';
	}
	snprintf(file_path, sizeof(file_path), "%s/%s", img_dir_path, fname);
	pr_info("GPU pages host path: %s\n", file_path);

	if (bind_mount_in_container(pid, file_path, gpu_restore_tmp_path) < 0) {
		pr_err("Failed to bind mount gpu-pages file into container\n");
		goto out;
	}

	/* Write the container-side path into the target's stack. */
	{
		struct user_regs_struct regs;
		struct iovec local_iov, remote_iov;
		const char *inject_path = gpu_restore_tmp_path;
		size_t path_len = strlen(inject_path) + 1;

		if (ptrace(PTRACE_GETREGS, tid, NULL, &regs) < 0) {
			pr_perror("PTRACE_GETREGS failed");
			goto out_umount;
		}
		path_addr = regs.rsp - 256;

		local_iov.iov_base = (void *)inject_path;
		local_iov.iov_len = path_len;
		remote_iov.iov_base = (void *)(uintptr_t)path_addr;
		remote_iov.iov_len = path_len;
		if (syscall(SYS_process_vm_writev, (pid_t)tid, &local_iov, 1UL,
			    &remote_iov, 1UL, 0UL) != (ssize_t)path_len) {
			pr_perror("process_vm_writev path failed");
			goto out_umount;
		}

		/*
		 * Try O_DIRECT first.  If the filesystem rejects it (EINVAL),
		 * fall back to buffered I/O — the pread loop below works either
		 * way, just at page-cache speed instead of NVMe DMA speed.
		 */
		target_fd = inject_syscall(tid, syscall_addr, SYS_openat,
					   (long)AT_FDCWD, (long)path_addr,
					   O_RDONLY | O_DIRECT, 0, 0, 0);
		if (target_fd == -EINVAL) {
			pr_info("O_DIRECT not supported, falling back to buffered I/O\n");
			target_fd = inject_syscall(tid, syscall_addr, SYS_openat,
						   (long)AT_FDCWD, (long)path_addr,
						   O_RDONLY, 0, 0, 0);
		}
	}

	if (target_fd < 0) {
		pr_err("openat injection failed: %ld\n", target_fd);
		goto out_umount;
	}

	/*
	 * For each GPU VMA: mlock first to pre-fault all anonymous pages
	 * (zero-fills them and pins them in RAM), then inject O_DIRECT pread64
	 * in chunks.  Pre-faulting is the key: without it, get_user_pages()
	 * inside the O_DIRECT path allocates and zero-fills pages on every DMA
	 * setup, limiting throughput to ~1.5 GB/s.  With pages already present
	 * and pinned, get_user_pages() is near-free and the NVMe controller can
	 * DMA at full sequential read bandwidth (~2.5 GB/s on this instance).
	 */
	t0 = now_ms();
	file_offset = GPU_PAGES_DATA_OFFSET;
	{
		double mlock_ms = 0, pread_ms = 0;
		double t1;

		for (i = 0; i < hdr.num_regions; i++) {
			uint64_t region_done = 0;

			/* THP hint: use 2MB pages when mlock faults them in (safe: does not set VM_HUGETLB) */
			inject_syscall(tid, syscall_addr, SYS_madvise,
				       (long)regions[i].start, (long)regions[i].size,
				       MADV_HUGEPAGE, 0, 0, 0);
			/* Pre-fault pages so O_DIRECT DMA doesn't pay fault overhead */
			t1 = now_ms();
			inject_syscall(tid, syscall_addr, SYS_mlock,
				       (long)regions[i].start, (long)regions[i].size,
				       0, 0, 0, 0);
			mlock_ms += now_ms() - t1;

			t1 = now_ms();
			while (region_done < regions[i].size) {
				uint64_t chunk = regions[i].size - region_done;
				long n;

				if (chunk > GPU_IO_CHUNK_SIZE)
					chunk = GPU_IO_CHUNK_SIZE;

				n = inject_syscall(tid, syscall_addr, SYS_pread64,
						   target_fd,
						   (long)(regions[i].start + region_done),
						   (long)chunk,
						   (long)(file_offset + region_done),
						   0, 0);
				if (n <= 0) {
					pr_err("pread64 injection failed for region %u at offset %llu: %ld\n",
					       i, (unsigned long long)region_done, n);
					inject_syscall(tid, syscall_addr, SYS_close,
						       target_fd, 0, 0, 0, 0, 0);
					goto out_umount;
				}
				region_done  += (uint64_t)n;
				total_bytes  += (uint64_t)n;
			}
			pread_ms += now_ms() - t1;

			/*
			 * Release VM_LOCKED before cuda-checkpoint restore runs.
			 * madvise(MADV_DONTNEED) returns EINVAL on VM_LOCKED pages
			 * (see can_madv_lru_vma()), which breaks the CUDA RM's
			 * post-restore staging buffer cleanup and leaves the context
			 * spinning on NV_ESC_RM_CONTROL.  Pages stay warm in RAM
			 * (recently written) so cuda-checkpoint still reads at full speed.
			 */
			inject_syscall(tid, syscall_addr, SYS_munlock,
				       (long)regions[i].start, (long)regions[i].size,
				       0, 0, 0, 0);

			file_offset += regions[i].size;
		}

		{
			double total_ms = now_ms() - t0;

			pr_info("[timing] O_DIRECT pread restore: %.0f ms (%.1f GB/s) [mlock=%.0f ms pread=%.0f ms]\n",
				total_ms, (double)total_bytes / total_ms / 1e6,
				mlock_ms, pread_ms);
		}
	}

	inject_syscall(tid, syscall_addr, SYS_close, target_fd, 0, 0, 0, 0, 0);
	pr_info("Loaded %u GPU regions for pid %d via O_DIRECT pread + mlock\n",
		hdr.num_regions, pid);
	ret = 0;
out_umount:
	umount_in_container(pid, gpu_restore_tmp_path);
out:
	free(regions);
	if (img_fd >= 0)
		close(img_fd);
	return ret;
}
