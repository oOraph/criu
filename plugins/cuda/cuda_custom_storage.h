#ifndef CUDA_CUSTOM_STORAGE_H
#define CUDA_CUSTOM_STORAGE_H

#include <stdbool.h>
#include <stddef.h>

#include "cuda_checkpoint.h"

/*
 * CUDA 13.4 (driver >= R615) custom-storage checkpoint/restore: the driver maps the target
 * process's GPU memory into the calling process (one contiguous region per GPU) and the caller
 * moves the bytes itself, then calls cuCheckpointOperationComplete().  No VRAM-sized host
 * staging in the target, and the copies run at whatever rate the storage and PCIe allow.
 */
typedef struct CUIcheckpointOperation_st *CUcheckpointOperationHandle;

typedef struct {
	unsigned long long devPtr; /* CUdeviceptr: zero-copy mapped device memory */
	size_t size;
	void *stream; /* CUstream in the primary context of that GPU */
} CUcheckpointCustomStoragePerDeviceData;

struct CUcheckpointCustomStorageInfo_st {
	CUcheckpointOperationHandle handle;
	CUcheckpointCustomStoragePerDeviceData *perDeviceData;
	unsigned int deviceCount;
};
typedef struct CUcheckpointCustomStorageInfo_st CUcheckpointCustomStorageInfo;

enum cuda_cs_mode {
	CUDA_CS_AUTO = 0, /* use it when the driver exposes the API */
	CUDA_CS_ON,	  /* require it */
	CUDA_CS_OFF,	  /* never */
};
extern enum cuda_cs_mode cuda_cs_mode;

/* Resolve the extra driver symbols from an already dlopen()ed libcuda; -ENOTSUP if the API is absent. */
int cuda_cs_init(void *libcuda_handle);
/* True when the API is available and the mode allows it: the dump uses custom storage. */
bool cuda_cs_active(void);
/* Fail unless a task checkpointed to custom storage can be restored from it. */
int cuda_cs_check_restore(int pid);
/* 1 if gpu-cs-<nspid>.img exists in img_dir_fd, 0 if not, -1 on error. */
int cuda_cs_image_exists(int pid, int img_dir_fd);
/* Remove a gpu-cs-<nspid>.img left by an earlier dump. */
int cuda_cs_image_remove(int pid, int img_dir_fd);
/* Retain the primary context of every device (required by the mode); call after cuInit(). */
int cuda_cs_prepare(void);
/*
 * Copy the mapped regions to (restore=false) or from (restore=true) gpu-cs-<nspid>.img in img_dir_fd.
 * On restore, each region gets the memory of the GPU that pairs (the device map, old to new UUID) maps
 * to its GPU, or of its own GPU without a pair.
 */
int cuda_cs_transfer(int pid, CUcheckpointCustomStorageInfo *info, int img_dir_fd, bool restore,
		     const CUcheckpointGpuPair *pairs, unsigned int npairs);
/*
 * The driver has no way to cancel a custom-storage checkpoint: once it is
 * completed, the GPU memory exists only where CRIU copied it. If the image
 * cannot be written, keep a copy in CRIU's memory so that the dump rollback
 * can still restore the task.
 */
struct cuda_cs_rescue;
struct cuda_cs_rescue *cuda_cs_rescue_save(CUcheckpointCustomStorageInfo *info);
int cuda_cs_rescue_restore(struct cuda_cs_rescue *r, CUcheckpointCustomStorageInfo *info);
void cuda_cs_rescue_free(struct cuda_cs_rescue *r);
/* Tell the driver the copies are done (synchronises its streams, unmaps). */
int cuda_cs_complete(CUcheckpointOperationHandle handle);

#endif /* CUDA_CUSTOM_STORAGE_H */
