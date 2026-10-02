/*
 * CUDA custom-storage checkpoint/restore engine (CUDA 13.4 driver API, driver >= R615).
 *
 * The Driver API backend calls cuCheckpointProcessCheckpoint()/Restore() with
 * customStorageInfo_out set; the driver then maps the target's GPU memory into
 * CRIU (one contiguous region per GPU, with a stream in that GPU's primary
 * context) and this file moves the bytes between those regions and
 * gpu-cs-<nspid>.img with N worker threads, each owning a CUDA stream and two
 * pinned 64 MB buffers (disk I/O of one chunk overlaps the PCIe transfer of the
 * other).  cuda_cs_complete() then lets the driver finish the operation.
 *
 * Notes: driver symbols are resolved through cuGetProcAddress, since dlsym
 * returns the legacy ABI of versioned symbols (CUDA_ERROR_INVALID_CONTEXT on
 * memcpy); for cuStreamGetCtx that is the 3-argument cuStreamGetCtx_v2 (the
 * third returns a green context, NULL for the primary contexts used here).
 * The caller must be allowed to ptrace the target, as CRIU already is.
 * The mapped pointer carries no CU_POINTER_ATTRIBUTE_CONTEXT; use the stream's.
 */
#include "criu-log.h"
#include "cuda_custom_storage.h"

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#ifdef LOG_PREFIX
#undef LOG_PREFIX
#endif
#define LOG_PREFIX "cuda_cs: "

#define CS_CHUNK  (64UL << 20)
#define CS_MAXTHR 32
#define CS_HDR	  4096
#define CS_MAGIC  0x43554353 /* "CUCS" */
#define CS_MAXDEV 32

enum cuda_cs_mode cuda_cs_mode = CUDA_CS_AUTO;
static bool cs_available;
static bool cs_prepared;

typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUstream;
typedef void *CUevent;
typedef unsigned long long CUdeviceptr;

static struct {
	CUresult (*get_proc_address)(const char *, void **, int, unsigned long long, int *);
	CUresult (*get_error_string)(CUresult, const char **);
	CUresult (*device_get_count)(int *);
	CUresult (*device_get)(CUdevice *, int);
	CUresult (*primary_ctx_retain)(CUcontext *, CUdevice);
	CUresult (*ctx_set_current)(CUcontext);
	CUresult (*stream_get_ctx)(CUstream, CUcontext *, void ** /* CUgreenCtx */);
	CUresult (*stream_create)(CUstream *, unsigned);
	CUresult (*stream_destroy)(CUstream);
	CUresult (*stream_synchronize)(CUstream);
	CUresult (*mem_host_alloc)(void **, size_t, unsigned);
	CUresult (*mem_free_host)(void *);
	CUresult (*memcpy_dtoh_async)(void *, CUdeviceptr, size_t, CUstream);
	CUresult (*memcpy_htod_async)(CUdeviceptr, const void *, size_t, CUstream);
	CUresult (*event_create)(CUevent *, unsigned);
	CUresult (*event_record)(CUevent, CUstream);
	CUresult (*event_synchronize)(CUevent);
	CUresult (*operation_complete)(CUcheckpointOperationHandle);
} cs;

#define CS_CUDA_VERSION 13040 /* ABI version to request from cuGetProcAddress */

static const char *cs_err(CUresult r)
{
	const char *s = "?";
	if (cs.get_error_string)
		cs.get_error_string(r, &s);
	return s;
}

static void *cs_resolve(void *h, const char *name)
{
	void *fn = NULL;
	int q;

	if (cs.get_proc_address && cs.get_proc_address(name, &fn, CS_CUDA_VERSION, 0, &q) == CUDA_SUCCESS && fn)
		return fn;
	return dlsym(h, name);
}

int cuda_cs_init(void *h)
{
	cs_available = false;
	if (!h)
		return -EINVAL;
	cs.get_proc_address = dlsym(h, "cuGetProcAddress_v2");
	if (!cs.get_proc_address)
		cs.get_proc_address = dlsym(h, "cuGetProcAddress");
	cs.operation_complete = dlsym(h, "cuCheckpointOperationComplete");
	if (!cs.operation_complete) {
		pr_debug("libcuda has no cuCheckpointOperationComplete: custom storage unavailable\n");
		return -ENOTSUP;
	}
#define R(field, name)                                                       \
	do {                                                                 \
		cs.field = cs_resolve(h, name);                              \
		if (!cs.field) {                                             \
			pr_err("Unable to resolve %s from libcuda\n", name); \
			return -ENOENT;                                      \
		}                                                            \
	} while (0)
	R(get_error_string, "cuGetErrorString");
	R(device_get_count, "cuDeviceGetCount");
	R(device_get, "cuDeviceGet");
	R(primary_ctx_retain, "cuDevicePrimaryCtxRetain");
	R(ctx_set_current, "cuCtxSetCurrent");
	R(stream_create, "cuStreamCreate");
	R(stream_destroy, "cuStreamDestroy");
	R(stream_synchronize, "cuStreamSynchronize");
	R(mem_host_alloc, "cuMemHostAlloc");
	R(mem_free_host, "cuMemFreeHost");
	R(memcpy_dtoh_async, "cuMemcpyDtoHAsync");
	R(memcpy_htod_async, "cuMemcpyHtoDAsync");
	R(event_create, "cuEventCreate");
	R(event_record, "cuEventRecord");
	R(event_synchronize, "cuEventSynchronize");
#undef R
	/* only the 3-argument ABI: never fall back to dlsym("cuStreamGetCtx"), the 2-argument one */
	{
		void *fn = NULL;
		int q;

		if (!cs.get_proc_address ||
		    cs.get_proc_address("cuStreamGetCtx", &fn, CS_CUDA_VERSION, 0, &q) != CUDA_SUCCESS || !fn)
			fn = dlsym(h, "cuStreamGetCtx_v2");
		if (!fn) {
			pr_err("Unable to resolve cuStreamGetCtx_v2 from libcuda\n");
			return -ENOENT;
		}
		cs.stream_get_ctx = fn;
	}
	cs_available = true;
	pr_info("custom-storage checkpoint API available\n");
	return 0;
}

bool cuda_cs_active(void)
{
	return cs_available && cuda_cs_mode != CUDA_CS_OFF;
}

int cuda_cs_prepare(void)
{
	int n, i;
	CUresult r;

	if (cs_prepared)
		return 0;
	r = cs.device_get_count(&n);
	if (r != CUDA_SUCCESS) {
		pr_err("cuDeviceGetCount: %s\n", cs_err(r));
		return -1;
	}
	for (i = 0; i < n; i++) {
		CUdevice d;
		CUcontext c;
		if ((r = cs.device_get(&d, i)) != CUDA_SUCCESS || (r = cs.primary_ctx_retain(&c, d)) != CUDA_SUCCESS) {
			pr_err("Unable to retain primary context of device %d: %s\n", i, cs_err(r));
			return -1;
		}
	}
	cs_prepared = true;
	return 0;
}

int cuda_cs_complete(CUcheckpointOperationHandle handle)
{
	CUresult r = cs.operation_complete(handle);
	if (r != CUDA_SUCCESS) {
		pr_err("cuCheckpointOperationComplete: %s\n", cs_err(r));
		return -1;
	}
	return 0;
}

static int cs_ns_pid(int pid)
{
	char path[64], line[256];
	int ns = pid;
	FILE *f;

	snprintf(path, sizeof(path), "/proc/%d/status", pid);
	f = fopen(path, "r");
	if (!f)
		return pid;
	while (fgets(line, sizeof(line), f)) {
		if (!strncmp(line, "NSpid:", 6)) {
			char *p = line + 6, *last = NULL, *tok;
			for (tok = strtok(p, " \t\n"); tok; tok = strtok(NULL, " \t\n"))
				last = tok;
			if (last)
				ns = atoi(last);
			break;
		}
	}
	fclose(f);
	return ns;
}

static double cs_now_ms(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1e3 + tv.tv_usec / 1e3;
}

static int cs_threads(void)
{
	const char *e = getenv("CUDA_CS_THREADS");
	int n = e ? atoi(e) : 0;
	/* Copies into the custom-storage mapping slow down with many concurrent streams, so default
	 * to a few threads; more threads only help when the storage is the bottleneck. */
	if (n <= 0)
		n = 4;
	if (n > CS_MAXTHR)
		n = CS_MAXTHR;
	return n < 1 ? 1 : n;
}

struct cs_xfer {
	int fd;
	off_t file_off;
	CUdeviceptr dptr;
	size_t size;
	CUcontext ctx;
	bool restore, direct;
	size_t nchunks;
	atomic_size_t next;
	atomic_int err;
};

struct cs_warg {
	struct cs_xfer *x;
	int id;
};

static void *cs_worker(void *p)
{
	struct cs_warg *w = p;
	struct cs_xfer *x = w->x;
	void *buf[2] = { NULL, NULL };
	CUevent ev[2] = { NULL, NULL };
	CUstream st = NULL;
	int inflight[2] = { 0, 0 };
	CUresult r;
	int b = 0, i;

	if ((r = cs.ctx_set_current(x->ctx)) != CUDA_SUCCESS) {
		pr_err("[w%d] cuCtxSetCurrent: %s\n", w->id, cs_err(r));
		goto fail;
	}
	if ((r = cs.stream_create(&st, 1 /* CU_STREAM_NON_BLOCKING */)) != CUDA_SUCCESS) {
		pr_err("[w%d] cuStreamCreate: %s\n", w->id, cs_err(r));
		goto fail;
	}
	for (i = 0; i < 2; i++) {
		if ((r = cs.mem_host_alloc(&buf[i], CS_CHUNK, 1 /* CU_MEMHOSTALLOC_PORTABLE */)) != CUDA_SUCCESS) {
			pr_err("[w%d] cuMemHostAlloc: %s\n", w->id, cs_err(r));
			goto fail;
		}
		if ((r = cs.event_create(&ev[i], 2 /* CU_EVENT_DISABLE_TIMING */)) != CUDA_SUCCESS) {
			pr_err("[w%d] cuEventCreate: %s\n", w->id, cs_err(r));
			goto fail;
		}
	}

	for (;; b ^= 1) {
		size_t k = atomic_fetch_add(&x->next, 1), off, len, iolen;

		if (k >= x->nchunks || atomic_load(&x->err))
			break;
		off = k * CS_CHUNK;
		len = x->size - off < CS_CHUNK ? x->size - off : CS_CHUNK;
		iolen = x->direct ? ((len + 4095) & ~4095UL) : len;
		if (inflight[b]) {
			cs.event_synchronize(ev[b]);
			inflight[b] = 0;
		}
		if (!x->restore) {
			if ((r = cs.memcpy_dtoh_async(buf[b], x->dptr + off, len, st)) != CUDA_SUCCESS) {
				pr_err("[w%d] cuMemcpyDtoHAsync: %s\n", w->id, cs_err(r));
				goto fail;
			}
			cs.event_record(ev[b], st);
			cs.event_synchronize(ev[b]);
			if (pwrite(x->fd, buf[b], iolen, x->file_off + off) != (ssize_t)iolen) {
				pr_perror("[w%d] pwrite", w->id);
				goto fail;
			}
		} else {
			if (pread(x->fd, buf[b], iolen, x->file_off + off) < (ssize_t)len) {
				pr_perror("[w%d] pread", w->id);
				goto fail;
			}
			if ((r = cs.memcpy_htod_async(x->dptr + off, buf[b], len, st)) != CUDA_SUCCESS) {
				pr_err("[w%d] cuMemcpyHtoDAsync: %s\n", w->id, cs_err(r));
				goto fail;
			}
			cs.event_record(ev[b], st);
			inflight[b] = 1;
		}
	}
	cs.stream_synchronize(st);
	goto out;
fail:
	atomic_store(&x->err, 1);
out:
	if (st) {
		cs.stream_synchronize(st);
		cs.stream_destroy(st);
	}
	for (i = 0; i < 2; i++)
		if (buf[i])
			cs.mem_free_host(buf[i]);
	return NULL;
}

static int cs_xfer_region(int fd, off_t file_off, CUcheckpointCustomStoragePerDeviceData *d, bool restore, bool direct)
{
	struct cs_xfer x;
	struct cs_warg wa[CS_MAXTHR];
	pthread_t th[CS_MAXTHR];
	CUcontext ctx = NULL;
	void *green = NULL;
	CUresult r;
	double t0;
	int n, i;

	if ((r = cs.stream_get_ctx(d->stream, &ctx, &green)) != CUDA_SUCCESS) {
		pr_err("Unable to get the mapping's context: %s\n", cs_err(r));
		return -1;
	}
	if (!ctx) {
		pr_err("The mapping's stream has no regular context (green context %p)\n", green);
		return -1;
	}
	if ((r = cs.ctx_set_current(ctx)) != CUDA_SUCCESS) {
		pr_err("Unable to set the mapping's context: %s\n", cs_err(r));
		return -1;
	}
	memset(&x, 0, sizeof(x));
	x.fd = fd;
	x.file_off = file_off;
	x.dptr = d->devPtr;
	x.size = d->size;
	x.ctx = ctx;
	x.restore = restore;
	x.direct = direct;
	x.nchunks = (d->size + CS_CHUNK - 1) / CS_CHUNK;
	atomic_init(&x.next, 0);
	atomic_init(&x.err, 0);

	n = cs_threads();
	if ((size_t)n > x.nchunks)
		n = (int)x.nchunks;
	t0 = cs_now_ms();
	for (i = 0; i < n; i++) {
		wa[i].x = &x;
		wa[i].id = i;
		if (pthread_create(&th[i], NULL, cs_worker, &wa[i])) {
			pr_perror("pthread_create");
			atomic_store(&x.err, 1);
			n = i;
			break;
		}
	}
	for (i = 0; i < n; i++)
		pthread_join(th[i], NULL);
	cs.stream_synchronize(d->stream);
	pr_info("[timing] custom-storage %s: %.2f GB, %d threads, %.0f ms (%.1f GB/s, %s)\n",
		restore ? "restore copy" : "checkpoint copy", d->size / 1e9, n, cs_now_ms() - t0,
		d->size / (cs_now_ms() - t0) / 1e6, direct ? "O_DIRECT" : "buffered");
	return atomic_load(&x.err) ? -1 : 0;
}

struct cs_hdr {
	uint32_t magic;
	uint32_t ndev;
	uint64_t size[CS_MAXDEV];
};

int cuda_cs_transfer(int pid, CUcheckpointCustomStorageInfo *info, int img_dir_fd, bool restore)
{
	char fname[64];
	int fd, flags = restore ? O_RDONLY : (O_WRONLY | O_CREAT | O_TRUNC);
	bool direct = true;
	struct cs_hdr h;
	void *hb = NULL;
	off_t off = CS_HDR;
	unsigned i;
	int ret = -1;

	if (!info || info->deviceCount > CS_MAXDEV) {
		pr_err("Bad custom storage info for pid %d\n", pid);
		return -1;
	}
	if (img_dir_fd < 0) {
		pr_err("No image directory for the custom-storage image of pid %d\n", pid);
		return -1;
	}
	snprintf(fname, sizeof(fname), "gpu-cs-%d.img", cs_ns_pid(pid));
	fd = openat(img_dir_fd, fname, flags | O_DIRECT, 0600);
	if (fd < 0 && errno == EINVAL) {
		direct = false;
		fd = openat(img_dir_fd, fname, flags, 0600);
	}
	if (fd < 0) {
		pr_perror("Unable to open %s", fname);
		return -1;
	}
	if (posix_memalign(&hb, 4096, CS_HDR))
		goto out;
	memset(hb, 0, CS_HDR);

	if (!restore) {
		memset(&h, 0, sizeof(h));
		h.magic = CS_MAGIC;
		h.ndev = info->deviceCount;
		for (i = 0; i < info->deviceCount; i++)
			h.size[i] = info->perDeviceData[i].size;
		memcpy(hb, &h, sizeof(h));
		if (pwrite(fd, hb, CS_HDR, 0) != CS_HDR) {
			pr_perror("Unable to write %s header", fname);
			goto out;
		}
	} else {
		if (pread(fd, hb, CS_HDR, 0) != CS_HDR) {
			pr_perror("Unable to read %s header", fname);
			goto out;
		}
		memcpy(&h, hb, sizeof(h));
		if (h.magic != CS_MAGIC || h.ndev != info->deviceCount) {
			pr_err("%s: bad magic or device count (image %u, driver %u)\n", fname, h.ndev, info->deviceCount);
			goto out;
		}
	}

	for (i = 0; i < info->deviceCount; i++) {
		CUcheckpointCustomStoragePerDeviceData *d = &info->perDeviceData[i];

		if (restore && d->size != h.size[i]) {
			pr_err("%s: device %u size mismatch (image %llu, driver %zu)\n", fname, i,
			       (unsigned long long)h.size[i], d->size);
			goto out;
		}
		if (cs_xfer_region(fd, off, d, restore, direct))
			goto out;
		off += (d->size + 4095) & ~4095UL;
	}
	if (!restore && !direct)
		fdatasync(fd);
	ret = 0;
out:
	free(hb);
	close(fd);
	return ret;
}
