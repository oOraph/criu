# CUDA Restore Slowdown: N GPUs vs 1 GPU

## Issue

Restore is significantly slower on a machine with N GPUs compared to a machine
with 1 GPU, even when the process being restored is in a cgroup that limits its
GPU visibility to 1 GPU.

## Root Cause

The CUDA plugin (`plugins/cuda/cuda_plugin.c`) restores GPU state by spawning
`cuda-checkpoint` as a subprocess via `fork`+`execvp` in `launch_cuda_checkpoint()`.
This happens multiple times per restore:

1. `cuda-checkpoint -h` — flag support check at plugin init
2. `cuda-checkpoint --get-restore-tid --pid <pid>`
3. `cuda-checkpoint --action restore --pid <pid>`
4. `cuda-checkpoint --action unlock --pid <pid>`

Each invocation calls `cuInit()` internally (standard NVIDIA runtime behavior),
which probes and initializes **all GPUs visible to the process**.

CRIU and its forked `cuda-checkpoint` children run **outside** the target
process's cgroup. They inherit the host's full GPU visibility (N GPUs), not the
1-GPU view the cgroup enforces on the target process. With N=4, each
`cuda-checkpoint` call is ~4x slower than with N=1, and there are 4+ calls per
restore.

## Fixes

### Option A: Set CUDA_VISIBLE_DEVICES before execvp (code change)

In `launch_cuda_checkpoint()`, set `CUDA_VISIBLE_DEVICES=<index>` in the
environment before `execvp`. This constrains `cuInit()` to a single GPU.

The GPU index must be derived from the dumped image or the target process's
cgroup device allow-list before restore begins.

### Option B: Spawn CRIU inside the GPU cgroup (operational change, no code change)

If CRIU itself is launched inside the cgroup that limits GPU visibility to 1 GPU,
all forked `cuda-checkpoint` children inherit that cgroup and only see 1 GPU.
`cuInit()` is fast. No code changes required, and no need to know the GPU index
explicitly.

This is the simpler fix if you control the launcher (e.g. zeropod/containerd
shim) and can place CRIU in the right cgroup before invoking it.
