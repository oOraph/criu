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

#ifndef SYS_process_vm_readv
#define SYS_process_vm_readv 310
#endif

#ifndef SYS_process_vm_writev
#define SYS_process_vm_writev 311
#endif

#ifndef SYS_mlock
#define SYS_mlock 149
#endif

#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif

#define pr_info(fmt, ...)   fprintf(stderr, "cuda_gpu_pages: " fmt, ##__VA_ARGS__)
#define pr_warn(fmt, ...)   fprintf(stderr, "cuda_gpu_pages: WARNING: " fmt, ##__VA_ARGS__)
#define pr_err(fmt, ...)    fprintf(stderr, "cuda_gpu_pages: ERROR: " fmt, ##__VA_ARGS__)
#define pr_perror(fmt, ...) fprintf(stderr, "cuda_gpu_pages: ERROR: " fmt ": %s\n", \
				    ##__VA_ARGS__, strerror(errno))

double now_ms(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
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

	hdr.magic = GPU_PAGES_MAGIC;
	hdr.num_regions = (uint32_t)count;

	snprintf(fname, sizeof(fname), "gpu-pages-%d.img", pid);
	fd = openat(img_dir_fd, fname, O_WRONLY | O_CREAT | O_TRUNC, 0600);
	if (fd < 0) {
		pr_perror("Cannot create %s", fname);
		return -1;
	}

	if (write(fd, &hdr, sizeof(hdr)) != (ssize_t)sizeof(hdr)) {
		pr_perror("Cannot write header to %s", fname);
		goto out;
	}

	if (count > 0 && write(fd, regions, (size_t)count * sizeof(*regions)) !=
				    (ssize_t)((size_t)count * sizeof(*regions))) {
		pr_perror("Cannot write region table to %s", fname);
		goto out;
	}

	/* Seek to page-aligned offset so page data can be mmap'd directly */
	if (lseek(fd, GPU_PAGES_DATA_OFFSET, SEEK_SET) != GPU_PAGES_DATA_OFFSET) {
		pr_perror("lseek to data offset failed");
		goto out;
	}

	buf = malloc(GPU_IO_CHUNK_SIZE);
	if (!buf) {
		pr_err("OOM: cannot allocate IO buffer\n");
		goto out;
	}

	for (i = 0; i < count; i++) {
		uint64_t offset = 0;
		uint64_t remaining = regions[i].size;

		while (remaining > 0) {
			size_t chunk = (remaining > GPU_IO_CHUNK_SIZE) ? GPU_IO_CHUNK_SIZE : (size_t)remaining;
			struct iovec local_iov = { .iov_base = buf, .iov_len = chunk };
			struct iovec remote_iov = { .iov_base = (void *)(uintptr_t)(regions[i].start + offset),
						    .iov_len = chunk };
			ssize_t n, written = 0;

			n = (ssize_t)syscall(SYS_process_vm_readv, (pid_t)pid, &local_iov, 1UL, &remote_iov, 1UL, 0UL);
			if (n < 0) {
				pr_perror("process_vm_readv failed for pid %d at 0x%lx", pid,
					  (unsigned long)(regions[i].start + offset));
				goto out;
			}

			while (written < n) {
				ssize_t w = write(fd, buf + written, (size_t)(n - written));

				if (w < 0) {
					pr_perror("write to %s failed", fname);
					goto out;
				}
				written += w;
			}
			offset += (uint64_t)n;
			remaining -= (uint64_t)n;
		}
	}

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
 * Restore GPU pages into the target process via file-backed mmap + mlock.
 *
 * For each GPU VMA we inject:
 *   mmap(addr, size, PROT_READ, MAP_FIXED|MAP_SHARED, fd, file_offset)
 *   madvise(MADV_POPULATE_READ)
 *   mlock(addr, size)
 *
 * MAP_FIXED|MAP_SHARED replaces the existing anonymous VMA with a
 * shared file-backed mapping. MAP_SHARED (not MAP_PRIVATE) avoids COW
 * page copies during MADV_POPULATE_READ — the kernel maps the existing
 * file pages directly without allocating new ones.  On warm storage
 * (tmpfs/page-cache) MADV_POPULATE_READ then completes near-instantly.
 *
 * After mlock the pages are resident and pinned; cuCheckpointProcessRestore
 * uses a faster (partial DMA) path when reading from locked pages.
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
	uint64_t file_offset;
	uint32_t i;
	long target_fd;
	uint64_t path_addr;

	snprintf(fname, sizeof(fname), "gpu-pages-%d.img", pid);
	img_fd = openat(img_dir_fd, fname, O_RDONLY);
	if (img_fd < 0) {
		if (errno == ENOENT) {
			pr_info("No gpu-pages file for pid %d, skipping\n", pid);
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
	 * Resolve the image file's absolute path so the target process can
	 * open it via its own openat() — the fd lives in the target's fd table.
	 */
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
	pr_info("GPU pages file: %s\n", file_path);

	/* Write path into target's stack (below the 128-byte x86-64 ABI red zone) */
	{
		struct user_regs_struct regs;
		struct iovec local_iov, remote_iov;
		size_t path_len = strlen(file_path) + 1;

		if (ptrace(PTRACE_GETREGS, tid, NULL, &regs) < 0) {
			pr_perror("PTRACE_GETREGS failed");
			goto out;
		}
		path_addr = regs.rsp - 256;

		local_iov.iov_base = file_path;
		local_iov.iov_len = path_len;
		remote_iov.iov_base = (void *)(uintptr_t)path_addr;
		remote_iov.iov_len = path_len;
		if (syscall(SYS_process_vm_writev, (pid_t)tid, &local_iov, 1UL,
			    &remote_iov, 1UL, 0UL) != (ssize_t)path_len) {
			pr_perror("process_vm_writev path failed");
			goto out;
		}

		target_fd = inject_syscall(tid, syscall_addr, SYS_openat,
					   (long)AT_FDCWD, (long)path_addr,
					   O_RDONLY, 0, 0, 0);
	}

	if (target_fd < 0) {
		pr_err("openat injection failed: %ld\n", target_fd);
		goto out;
	}

	/*
	 * For each GPU VMA: remap as MAP_FIXED|MAP_SHARED file-backed, then
	 * use MADV_POPULATE_READ to fault all pages in from disk synchronously
	 * in one sequential pass (no RLIMIT_MEMLOCK needed, unlike mlock).
	 * Then attempt mlock as best-effort to pin pages for DMA.
	 *
	 * MAP_SHARED (not MAP_PRIVATE) avoids COW page copies: with MAP_PRIVATE
	 * MADV_POPULATE_READ triggers copy-on-write allocation for every page
	 * (~1.5 GB/s, same cost as a CPU copy).  MAP_SHARED maps the existing
	 * file pages directly (zero data movement) — MADV_POPULATE_READ is then
	 * near-instant on warm storage (tmpfs/page-cache).  cuda-checkpoint only
	 * reads these pages (copies to VRAM), so shared visibility is safe.
	 *
	 * File offsets and VMA addresses are always page-aligned (kernel VMAs),
	 * and GPU_PAGES_DATA_OFFSET=4096 is one page — mmap offset constraint
	 * (must be page-aligned) is always satisfied.
	 */
	file_offset = GPU_PAGES_DATA_OFFSET;
	for (i = 0; i < hdr.num_regions; i++) {
		long mapped = inject_syscall(tid, syscall_addr, SYS_mmap,
					     (long)regions[i].start,
					     (long)regions[i].size,
					     PROT_READ,
					     MAP_FIXED | MAP_SHARED,
					     target_fd,
					     (long)file_offset);
		if (mapped != (long)regions[i].start) {
			pr_err("mmap injection failed for region %u: got 0x%lx, expected 0x%llx\n",
			       i, mapped, (unsigned long long)regions[i].start);
			inject_syscall(tid, syscall_addr, SYS_close, target_fd, 0, 0, 0, 0, 0);
			goto out;
		}

		/*
		 * MADV_POPULATE_READ faults all pages in synchronously from the
		 * file — no RLIMIT_MEMLOCK needed.  Pages are in RAM after this.
		 * mlock follows as best-effort to pin them for DMA (may fail if
		 * RLIMIT_MEMLOCK is too small; pages are still warm in RAM).
		 */
		inject_syscall(tid, syscall_addr, SYS_madvise,
			       (long)regions[i].start, (long)regions[i].size,
			       MADV_POPULATE_READ, 0, 0, 0);
		inject_syscall(tid, syscall_addr, SYS_mlock,
			       (long)regions[i].start, (long)regions[i].size,
			       0, 0, 0, 0);

		file_offset += regions[i].size;
	}

	inject_syscall(tid, syscall_addr, SYS_close, target_fd, 0, 0, 0, 0, 0);
	pr_info("Loaded %u GPU regions for pid %d via file-backed mmap + mlock\n",
		hdr.num_regions, pid);
	ret = 0;
out:
	free(regions);
	if (img_fd >= 0)
		close(img_fd);
	return ret;
}
