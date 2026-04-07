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

#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE 14
#endif

#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif

#ifndef SYS_sync_file_range
#define SYS_sync_file_range 277
#endif

#ifndef SYNC_FILE_RANGE_WRITE
#define SYNC_FILE_RANGE_WRITE 2
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
	/*
	 * Start async writeback of the image file so the NVMe write is in
	 * flight while the rest of the dump proceeds.  sync_file_range(WRITE)
	 * submits writeback I/O and returns immediately — unlike
	 * posix_fadvise(DONTNEED) which also calls invalidate_mapping_pages()
	 * and scans 1.5M pages even for dirty ones, adding ~4s of overhead.
	 * By the time restore calls O_DIRECT pread, the write is likely done
	 * and the page cache can be evicted cheaply.
	 */
	syscall(SYS_sync_file_range, fd, (int64_t)GPU_PAGES_DATA_OFFSET, (int64_t)0,
		SYNC_FILE_RANGE_WRITE);
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
	 * open it via its own openat().
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

	/*
	 * Evict the image file's page cache before O_DIRECT pread.
	 * dump_gpu_pages() called sync_file_range(WRITE) to start async NVMe
	 * writeback; by now the write is likely done so pages are clean.
	 * fadvise(DONTNEED) on clean pages is fast (~300ms for 6 GB) vs ~4s
	 * on dirty ones.  Without this, O_DIRECT calls
	 * invalidate_inode_pages2_range() per 64 MB chunk and waits for any
	 * remaining dirty pages, serialising read and write I/O.
	 */
	{
		int fadvise_fd = open(file_path, O_RDONLY);
		double t_fadvise = now_ms();

		if (fadvise_fd >= 0) {
			posix_fadvise(fadvise_fd, GPU_PAGES_DATA_OFFSET, 0,
				      POSIX_FADV_DONTNEED);
			close(fadvise_fd);
		}
		pr_info("[timing] pre-pread fadvise: %.0f ms\n", now_ms() - t_fadvise);
	}

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
		goto out;
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

			/*
			 * Replace the VMA with MAP_HUGETLB to get guaranteed 2MB pages.
			 * Unlike MADV_HUGEPAGE (THP), MAP_HUGETLB allocates from the
			 * pre-reserved huge page pool and never falls back to 4KB pages.
			 * If it fails (pool empty or address/size not 2MB-aligned), re-
			 * create the VMA normally and fall back to MADV_HUGEPAGE.
			 */
			{
				long r = inject_syscall(tid, syscall_addr, SYS_mmap,
							(long)regions[i].start,
							(long)regions[i].size,
							PROT_READ | PROT_WRITE,
							MAP_PRIVATE | MAP_ANONYMOUS |
							MAP_FIXED | MAP_HUGETLB,
							-1L, 0L);

				if (r != (long)regions[i].start) {
					inject_syscall(tid, syscall_addr, SYS_mmap,
						       (long)regions[i].start,
						       (long)regions[i].size,
						       PROT_READ | PROT_WRITE,
						       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
						       -1L, 0L);
					inject_syscall(tid, syscall_addr, SYS_madvise,
						       (long)regions[i].start,
						       (long)regions[i].size,
						       MADV_HUGEPAGE, 0, 0, 0);
				}
			}

			/* Pre-fault + pin pages before the O_DIRECT read */
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
					goto out;
				}
				region_done  += (uint64_t)n;
				total_bytes  += (uint64_t)n;
			}
			pread_ms += now_ms() - t1;

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
out:
	free(regions);
	if (img_fd >= 0)
		close(img_fd);
	return ret;
}
