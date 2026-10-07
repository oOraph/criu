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
#include "cuda.pb-c.h"

#include <dlfcn.h>
#include <endian.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>

#ifdef LOG_PREFIX
#undef LOG_PREFIX
#endif
#define LOG_PREFIX "cuda_cs: "

#define CS_CHUNK  (64UL << 20)
#define CS_MAXTHR 32
#define CS_HDR	  4096
#define CS_MAGIC  "CUCS"
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

int cuda_cs_check_restore(int pid)
{
	if (!cs_available) {
		pr_err("pid %d was checkpointed to custom storage, but libcuda has no custom-storage checkpoint API\n",
		       pid);
		return -1;
	}
	if (cuda_cs_mode == CUDA_CS_OFF) {
		pr_err("pid %d was checkpointed to custom storage and cannot be restored with cuda_plugin.custom-storage=off\n",
		       pid);
		return -1;
	}
	return 0;
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

static void cs_image_name(int pid, char *buf, size_t len)
{
	snprintf(buf, len, "gpu-cs-%d.img", cs_ns_pid(pid));
}

int cuda_cs_image_exists(int pid, int img_dir_fd)
{
	char fname[64];

	cs_image_name(pid, fname, sizeof(fname));
	if (!faccessat(img_dir_fd, fname, F_OK, 0))
		return 1;
	if (errno == ENOENT)
		return 0;
	pr_perror("Unable to check for %s", fname);
	return -1;
}

int cuda_cs_image_remove(int pid, int img_dir_fd)
{
	char fname[64];

	cs_image_name(pid, fname, sizeof(fname));
	if (unlinkat(img_dir_fd, fname, 0) && errno != ENOENT) {
		pr_perror("Unable to remove stale %s", fname);
		return -1;
	}
	return 0;
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
	char *mem; /* host memory to copy to or from instead of fd */
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

/* A failed copy may only be reported by a later synchronisation: check every call. */
#define CS_CALL(call, name)                                                 \
	do {                                                                \
		CUresult __r = (call);                                      \
		if (__r != CUDA_SUCCESS) {                                  \
			pr_err("[w%d] %s: %s\n", w->id, name, cs_err(__r)); \
			goto fail;                                          \
		}                                                           \
	} while (0)

/* Wait for the copy of a chunk into buf, then store it. */
static int cs_store(struct cs_warg *w, void *buf, CUevent ev, size_t off, size_t len)
{
	struct cs_xfer *x = w->x;
	size_t iolen = x->direct ? ((len + 4095) & ~4095UL) : len;
	ssize_t done;

	CS_CALL(cs.event_synchronize(ev), "cuEventSynchronize");
	/* The O_DIRECT tail of the last chunk would hold the previous chunk's bytes. */
	memset((char *)buf + len, 0, iolen - len);
	if (x->mem) {
		memcpy(x->mem + x->file_off + off, buf, len);
	} else if ((done = pwrite(x->fd, buf, iolen, x->file_off + off)) != (ssize_t)iolen) {
		if (done < 0)
			pr_perror("[w%d] pwrite", w->id);
		else
			pr_err("[w%d] short write: %zd of %zu bytes\n", w->id, done, iolen);
		goto fail;
	}
	return 0;
fail:
	return -1;
}

static void *cs_worker(void *p)
{
	struct cs_warg *w = p;
	struct cs_xfer *x = w->x;
	void *buf[2] = { NULL, NULL };
	CUevent ev[2] = { NULL, NULL };
	CUstream st = NULL;
	int inflight[2] = { 0, 0 };
	size_t prev_off = 0, prev_len = 0;
	bool pending = false;
	ssize_t done;
	int b = 0, i;

	CS_CALL(cs.ctx_set_current(x->ctx), "cuCtxSetCurrent");
	CS_CALL(cs.stream_create(&st, 1 /* CU_STREAM_NON_BLOCKING */), "cuStreamCreate");
	for (i = 0; i < 2; i++) {
		CS_CALL(cs.mem_host_alloc(&buf[i], CS_CHUNK, 1 /* CU_MEMHOSTALLOC_PORTABLE */), "cuMemHostAlloc");
		CS_CALL(cs.event_create(&ev[i], 2 /* CU_EVENT_DISABLE_TIMING */), "cuEventCreate");
	}

	for (;; b ^= 1) {
		size_t k = atomic_fetch_add(&x->next, 1), off, len, iolen;

		if (k >= x->nchunks || atomic_load(&x->err))
			break;
		off = k * CS_CHUNK;
		len = x->size - off < CS_CHUNK ? x->size - off : CS_CHUNK;
		iolen = x->direct ? ((len + 4095) & ~4095UL) : len;
		if (inflight[b]) {
			CS_CALL(cs.event_synchronize(ev[b]), "cuEventSynchronize");
			inflight[b] = 0;
		}
		if (!x->restore) {
			/* Copy this chunk from the GPU while the previous one, in the other buffer, is stored. */
			CS_CALL(cs.memcpy_dtoh_async(buf[b], x->dptr + off, len, st), "cuMemcpyDtoHAsync");
			CS_CALL(cs.event_record(ev[b], st), "cuEventRecord");
			if (pending && cs_store(w, buf[b ^ 1], ev[b ^ 1], prev_off, prev_len))
				goto fail;
			pending = true;
			prev_off = off;
			prev_len = len;
		} else {
			if (x->mem) {
				memcpy(buf[b], x->mem + x->file_off + off, len);
			} else if ((done = pread(x->fd, buf[b], iolen, x->file_off + off)) < (ssize_t)len) {
				if (done < 0)
					pr_perror("[w%d] pread", w->id);
				else
					pr_err("[w%d] short read: %zd of %zu bytes, the image is truncated\n", w->id, done,
					       len);
				goto fail;
			}
			CS_CALL(cs.memcpy_htod_async(x->dptr + off, buf[b], len, st), "cuMemcpyHtoDAsync");
			CS_CALL(cs.event_record(ev[b], st), "cuEventRecord");
			inflight[b] = 1;
		}
	}
	/* The loop toggled b after the last chunk: it is in the other buffer. */
	if (pending && cs_store(w, buf[b ^ 1], ev[b ^ 1], prev_off, prev_len))
		goto fail;
	CS_CALL(cs.stream_synchronize(st), "cuStreamSynchronize");
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
#undef CS_CALL

static int cs_xfer_region(int fd, char *mem, off_t file_off, CUcheckpointCustomStoragePerDeviceData *d, bool restore,
			  bool direct)
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
	x.mem = mem;
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
		errno = pthread_create(&th[i], NULL, cs_worker, &wa[i]);
		if (errno) {
			pr_perror("pthread_create");
			atomic_store(&x.err, 1);
			n = i;
			break;
		}
	}
	for (i = 0; i < n; i++)
		pthread_join(th[i], NULL);
	if ((r = cs.stream_synchronize(d->stream)) != CUDA_SUCCESS) {
		pr_err("cuStreamSynchronize on the mapping's stream: %s\n", cs_err(r));
		atomic_store(&x.err, 1);
	}
	pr_info("[timing] custom-storage %s: %.2f GB, %d threads, %.0f ms (%.1f GB/s, %s)\n",
		restore ? "restore copy" : "checkpoint copy", d->size / 1e9, n, cs_now_ms() - t0,
		d->size / (cs_now_ms() - t0) / 1e6, mem ? "host memory" : direct ? "O_DIRECT" : "buffered");
	return atomic_load(&x.err) ? -1 : 0;
}

#define CS_IMAGE_VERSION 1

static size_t cs_align(size_t n)
{
	return (n + 4095) & ~4095UL;
}

/* Lay the regions out after the header and write it. */
static int cs_write_header(int fd, void *hb, CUcheckpointCustomStorageInfo *info, CudaCsRegion *regions,
			   const char *fname)
{
	CudaCsRegion *rp[CS_MAXDEV];
	CudaCsImage img = CUDA_CS_IMAGE__INIT;
	uint32_t len32;
	size_t len;
	off_t off = CS_HDR;
	unsigned int i;

	for (i = 0; i < info->deviceCount; i++) {
		cuda_cs_region__init(&regions[i]);
		regions[i].has_size = regions[i].has_offset = true;
		regions[i].size = info->perDeviceData[i].size;
		regions[i].offset = off;
		off += cs_align(regions[i].size);
		rp[i] = &regions[i];
	}
	img.has_version = true;
	img.version = CS_IMAGE_VERSION;
	img.n_regions = info->deviceCount;
	img.regions = rp;
	len = cuda_cs_image__get_packed_size(&img);
	if (len > CS_HDR - 8) {
		pr_err("%s: header of %zu bytes does not fit\n", fname, len);
		return -1;
	}
	len32 = htole32(len);
	memcpy(hb, CS_MAGIC, 4);
	memcpy((char *)hb + 4, &len32, 4);
	cuda_cs_image__pack(&img, (uint8_t *)hb + 8);

	if (pwrite(fd, hb, CS_HDR, 0) != CS_HDR) {
		pr_perror("Unable to write %s header", fname);
		return -1;
	}
	return 0;
}

static CudaCsImage *cs_read_header(int fd, void *hb, const char *fname)
{
	CudaCsImage *img;
	uint32_t len;
	unsigned int i;
	ssize_t n;

	n = pread(fd, hb, CS_HDR, 0);
	if (n != CS_HDR) {
		if (n < 0)
			pr_perror("Unable to read %s header", fname);
		else
			pr_err("%s: truncated header\n", fname);
		return NULL;
	}
	memcpy(&len, (char *)hb + 4, 4);
	len = le32toh(len);
	if (memcmp(hb, CS_MAGIC, 4) || len > CS_HDR - 8) {
		pr_err("%s is not a custom-storage image\n", fname);
		return NULL;
	}
	img = cuda_cs_image__unpack(NULL, len, (uint8_t *)hb + 8);
	if (!img) {
		pr_err("Unable to unpack the %s header\n", fname);
		return NULL;
	}
	if (img->version != CS_IMAGE_VERSION) {
		pr_err("%s: unsupported version %u\n", fname, img->version);
		goto err;
	}
	for (i = 0; i < img->n_regions; i++) {
		if (!img->regions[i]->has_size || !img->regions[i]->has_offset ||
		    img->regions[i]->offset % 4096 || img->regions[i]->offset < CS_HDR) {
			pr_err("%s: invalid region %u\n", fname, i);
			goto err;
		}
	}
	return img;
err:
	cuda_cs_image__free_unpacked(img, NULL);
	return NULL;
}

int cuda_cs_transfer(int pid, CUcheckpointCustomStorageInfo *info, int img_dir_fd, bool restore)
{
	char fname[64];
	int fd, flags = restore ? O_RDONLY : (O_WRONLY | O_CREAT | O_TRUNC);
	bool direct = true;
	CudaCsRegion regions[CS_MAXDEV];
	CudaCsImage *img = NULL;
	void *hb = NULL;
	unsigned int i;
	int ret = -1;

	if (!info || info->deviceCount > CS_MAXDEV) {
		pr_err("Bad custom storage info for pid %d\n", pid);
		return -1;
	}
	if (img_dir_fd < 0) {
		pr_err("No image directory for the custom-storage image of pid %d\n", pid);
		return -1;
	}
	cs_image_name(pid, fname, sizeof(fname));
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
		if (cs_write_header(fd, hb, info, regions, fname))
			goto out;
	} else {
		img = cs_read_header(fd, hb, fname);
		if (!img)
			goto out;
		if (img->n_regions != info->deviceCount) {
			pr_err("%s holds %zu devices, the driver maps %u\n", fname, img->n_regions, info->deviceCount);
			goto out;
		}
	}

	for (i = 0; i < info->deviceCount; i++) {
		CUcheckpointCustomStoragePerDeviceData *d = &info->perDeviceData[i];
		CudaCsRegion *r = restore ? img->regions[i] : &regions[i];

		if (d->size != r->size) {
			pr_err("%s: device %u size mismatch (image %llu, driver %zu)\n", fname, i,
			       (unsigned long long)r->size, d->size);
			goto out;
		}
		if (cs_xfer_region(fd, NULL, r->offset, d, restore, direct))
			goto out;
	}
	/* O_DIRECT bypasses the page cache, not the device's write cache. */
	if (!restore && fdatasync(fd)) {
		pr_perror("Unable to sync %s", fname);
		goto out;
	}
	ret = 0;
out:
	if (img)
		cuda_cs_image__free_unpacked(img, NULL);
	free(hb);
	close(fd);
	return ret;
}

struct cuda_cs_rescue {
	char *mem;
	size_t len;
	unsigned int ndev;
	size_t size[CS_MAXDEV];
};

struct cuda_cs_rescue *cuda_cs_rescue_save(CUcheckpointCustomStorageInfo *info)
{
	struct cuda_cs_rescue *r;
	size_t off = 0;
	unsigned int i;

	if (!info || info->deviceCount > CS_MAXDEV)
		return NULL;
	r = calloc(1, sizeof(*r));
	if (!r)
		return NULL;
	r->ndev = info->deviceCount;
	for (i = 0; i < r->ndev; i++) {
		r->size[i] = info->perDeviceData[i].size;
		r->len += r->size[i];
	}
	r->mem = mmap(NULL, r->len ?: 1, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (r->mem == MAP_FAILED) {
		pr_perror("Unable to allocate %zu bytes to keep the GPU memory", r->len);
		free(r);
		return NULL;
	}
	for (i = 0; i < r->ndev; i++) {
		if (cs_xfer_region(-1, r->mem, off, &info->perDeviceData[i], false, false)) {
			cuda_cs_rescue_free(r);
			return NULL;
		}
		off += r->size[i];
	}
	pr_info("Kept %zu bytes of GPU memory in host memory for the rollback\n", r->len);
	return r;
}

int cuda_cs_rescue_restore(struct cuda_cs_rescue *r, CUcheckpointCustomStorageInfo *info)
{
	size_t off = 0;
	unsigned int i;

	if (!info || info->deviceCount != r->ndev) {
		pr_err("The driver maps %u devices to restore, %u were kept\n", info ? info->deviceCount : 0, r->ndev);
		return -1;
	}
	for (i = 0; i < r->ndev; i++) {
		if (info->perDeviceData[i].size != r->size[i]) {
			pr_err("Device %u: the driver maps %zu bytes to restore, %zu were kept\n", i,
			       info->perDeviceData[i].size, r->size[i]);
			return -1;
		}
		if (cs_xfer_region(-1, r->mem, off, &info->perDeviceData[i], true, false))
			return -1;
		off += r->size[i];
	}
	return 0;
}

void cuda_cs_rescue_free(struct cuda_cs_rescue *r)
{
	if (!r)
		return;
	munmap(r->mem, r->len ?: 1);
	free(r);
}
