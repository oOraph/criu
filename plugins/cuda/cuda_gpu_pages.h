/*
 * cuda_gpu_pages.h - shared GPU VRAM page I/O routines
 *
 * Shared between cuda_plugin.so (CRIU plugin) and cuda-offload (standalone).
 * Implementation is in cuda_gpu_pages.c.
 *
 * ---- Fast GPU page I/O ----
 *
 * After the CUDA checkpoint (cuda-checkpoint --action checkpoint or
 * cuCheckpointProcessCheckpoint), VRAM data is moved into new anonymous
 * private mappings ("staging VMAs") in the target process. These pages would
 * normally go through CRIU's slow ptrace page walk. Instead we:
 *
 *   Dump: scan /proc/<pid>/maps before/after checkpoint, diff to find the
 *         new staging VMAs, copy them out with process_vm_readv and write
 *         gpu-pages-<ns_pid>.img with O_DIRECT, then free them via injected
 *         madvise(MADV_DONTNEED) so CRIU sees empty pages and skips them.
 *         The VMAs themselves stay in the address space and CRIU restores
 *         them as ordinary (empty) anonymous mappings.
 *
 *   Restore: with the target stopped, bind-mount the image into its mount
 *            namespace, then inject syscalls into one of its threads:
 *            openat(O_DIRECT), and per staging VMA madvise(MADV_HUGEPAGE),
 *            mlock (pre-fault + pin so the DMA target pages exist), pread64
 *            in GPU_IO_CHUNK_SIZE chunks straight into the VMA, munlock (so
 *            the CUDA RM's later MADV_DONTNEED cleanup is not refused on
 *            VM_LOCKED pages), and finally close.  The backend's CUDA
 *            restore then reads the refilled pages back into VRAM.  Falls
 *            back to buffered reads when the filesystem rejects O_DIRECT.
 */

#ifndef CUDA_GPU_PAGES_H
#define CUDA_GPU_PAGES_H

#include <stdint.h>

#define GPU_PAGES_MAGIC		0x47505544u  /* "GPUD" */
#define GPU_IO_CHUNK_SIZE	(64 * 1024 * 1024)
/* Page data starts at this offset in the image file (page-aligned for O_DIRECT) */
#define GPU_PAGES_DATA_OFFSET	4096

struct gpu_region {
	uint64_t start;
	uint64_t size;
};

struct gpu_pages_hdr {
	uint32_t magic;
	uint32_t num_regions;
};

double now_ms(void);

int scan_anon_private_vmas(int pid, struct gpu_region **out, int *count);

int diff_anon_vmas(struct gpu_region *before, int n_before,
		   struct gpu_region *after, int n_after,
		   struct gpu_region **diff_out, int *diff_count);

int dump_gpu_pages(int pid, int img_dir_fd, struct gpu_region *regions, int count);

uint64_t find_syscall_addr(int pid);

long inject_syscall(int tid, uint64_t syscall_addr,
		    long nr, long a1, long a2, long a3,
		    long a4, long a5, long a6);

int inject_madvise_dontneed(int tid, uint64_t addr, uint64_t len, uint64_t syscall_addr);

int release_gpu_pages(int tid, uint64_t syscall_addr, struct gpu_region *regions, int count);

int restore_gpu_pages(int pid, int tid, uint64_t syscall_addr, int img_dir_fd);

#endif /* CUDA_GPU_PAGES_H */
