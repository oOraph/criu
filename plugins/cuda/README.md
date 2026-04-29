Checkpoint and Restore for CUDA applications with CRIU
======================================================

# Requirements
The cuda-checkpoint utility should be placed somewhere in your $PATH and an r555
or higher GPU driver is required for CUDA CRIU integration support.

## cuda-checkpoint
The cuda-checkpoint utility can be found at:
https://github.com/NVIDIA/cuda-checkpoint

cuda-checkpoint is a binary utility used to issue checkpointing commands to CUDA
applications. Updating the cuda-checkpoint utility between driver releases
should not be necessary as the utility simply exposes some extra driver behavior
so driver updates are all that's needed to get access to newer features.

# Checkpointing Procedure
cuda-checkpoint exposes 4 actions used in the checkpointing process: lock,
checkpoint, restore, unlock.

* lock - Used with the PAUSE_DEVICES hook while a process is still running to
  quiesce the application into a state where it can be checkpointed
* checkpoint - Used with the CHECKPOINT_DEVICES hook once a process has been
  seized/frozen to perform the actual checkpointing operation
* restore/unlock - Used with the RESUME_DEVICES_LATE hook to restore the CUDA
  state and release the process back to it's running state

These actions are facilitated by a CUDA checkpoint+restore thread that the CUDA
plugin will re-wake when needed.

# Known Limitations
* Currently GPU memory contents are brought into main system memory and CRIU
  then checkpoints that as part of the normal procedure. On systems with many
  GPU's with high GPU memory usage this can cause memory thrashing. A future
  CUDA release will add support for dumping the memory contents to files to
  alleviate this as well as support in the CRIU plugin.
* There's currently a small race between when a PAUSE_DEVICES hook is called on
  a running process and a process calls cuInit() and finishes initializing CUDA
  after the PAUSE is issued but before the process is frozen to checkpoint. This
  will cause cuda-checkpoint to report that the process is in an illegal state
  for checkpointing and it's recommended to just attempt the CRIU procedure
  again, this should be very rare.
* Applications that use NVML will leave some leftover device references as NVML
  is not currently supported for checkpointing. There will be support for this
  in later drivers. A possible temporary workaround is to have the
  {DUMP,RESTORE}_EXT_FILE hook just ignore /dev/nvidiactl and /dev/nvidia{0..N}
  remaining references for these applications as in most cases NVML is used to
  get info such as gpu count and some capabilities and these values are never
  accessed again and unlikely to change.
* CUDA applications that fork() but don't call exec() but also don't issue any
  CUDA API calls will have some leftover references to /dev/nvidia* and fail to
  checkpoint as a result. This can be worked around in a similar fashion to the
  NVML case where the leftover references can be ignored as CUDA is not fork()
  safe anyway.
* Restore currently requires that you restore on a system with similar GPU's and
  same GPU count.
* NVIDIA UVM Managed Memory, MIG (Multi Instance GPU), and MPS (Multi-Process
  Service) are currently not supported for checkpointing. Future CUDA releases
  will add support for these.

# cuda-offload: External Two-Step Checkpoint Tool

`cuda-offload` is a standalone binary that implements an alternative
checkpoint/restore flow where VRAM offload happens **outside** of CRIU, before
the dump and after the restore.  This avoids CRIU touching GPU pages at all and
allows using O_DIRECT I/O for the GPU image files.

This separation also enables **light vs. heavy** checkpoint/restore:
- **Heavy** (full migration): run both `cuda-offload checkpoint` + `criu dump`
  on one machine and `criu restore` + `cuda-offload restore` on another.
  VRAM pages travel with the CRIU image.
- **Light** (GPU-only preemption): run `cuda-offload checkpoint` to evict VRAM
  to disk and free the GPU, then `cuda-offload restore` later on the same
  machine — no `criu dump`/`restore` needed.  CPU state is never touched.

## Flow

**Checkpoint** (run before `criu dump`):
1. `cuda-offload --action checkpoint --pid PID` seizes the whole process tree
   with `PTRACE_SEIZE + PTRACE_INTERRUPT`, freezing every process.
2. For each process with a CUDA context: lock → resume restore thread →
   `cuda-checkpoint --action checkpoint` → re-freeze restore thread.
3. New anonymous VMAs created by the checkpoint (VRAM data now in CPU RAM) are
   identified by diffing `/proc/pid/maps` before and after, dumped to
   `gpu-pages-<pid>.img` with `process_vm_readv`, then freed with an injected
   `madvise(MADV_DONTNEED)` so CRIU skips them in its page walk.
4. Wall-clock timers (`ITIMER_REAL`, POSIX `CLOCK_REALTIME/MONOTONIC/BOOTTIME`)
   are **saved and disarmed** at seize time and written to
   `gpu-offload-timers-<pid>.img`.
5. The whole tree is detached with `SIGSTOP` so it stays frozen for `criu dump`.

**Restore** (run after `criu restore`):
1. `criu restore` brings back the process tree.  The CRIU cuda plugin detects
   `gpu-offload-external.marker` and returns immediately without calling
   `resume_device`, leaving the processes in the same stopped state they were
   dumped in (group-stop from the SIGSTOP in step 5 above).
2. `cuda-offload --action restore --pid PID` seizes the tree again and, for
   each process with a GPU image: injects `mmap(MAP_SHARED) + mlock` to reload
   pages from `gpu-pages-<pid>.img` into the process address space.
3. The restore thread is seized and resumed.  A `SIGCONT` is sent to the
   process to clear the group-stop that CRIU faithfully restored — without it
   the thread re-enters group-stop immediately after `PTRACE_CONT` and
   `cuda-checkpoint --action restore` hangs waiting for it.  This matches the
   `cuda_plugin.c` `RESUME_DEVICES_LATE` behaviour where the whole process is
   already running when restore is triggered.
4. `cuda-checkpoint --action restore` then `--action unlock` drive VRAM reload.
5. Wall-clock timers are **re-armed** from the saved image, corrected for the
   elapsed freeze time, overriding CRIU's own timer restore.
6. The tree is detached with `SIGCONT` to resume normal execution.

## Ptrace stop vs. group-stop after CRIU restore

CRIU restores process state exactly as it was at dump time.  Because the
process was dumped while in group-stop (SIGSTOP from step 5 of checkpoint),
after `criu restore` it is still in group-stop.  When cuda-offload seizes a
thread and calls `PTRACE_CONT`, the kernel group-stop state is still active and
the thread re-enters group-stop immediately instead of running.  The fix is to
call `kill(pid, SIGCONT)` right after `PTRACE_CONT`; threads already in
ptrace-stop (the main thread seized by cuda-offload's outer loop) are
unaffected.
