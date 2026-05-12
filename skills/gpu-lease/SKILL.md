---
name: gpu-lease
description: Use before running local GPU workloads such as PyTorch, SGLang serving, Ray clusters, CUDA benchmarks, or scripts that use GPUs; instructs agents to wrap commands with gpu-lease run so CUDA_VISIBLE_DEVICES is set through a lease.
---

# GPU Lease

Use this skill before running local GPU workloads from Codex or another code agent.

GPU workloads include PyTorch training or inference, SGLang serving, Ray workers or clusters,
CUDA benchmarks, and scripts that import GPU frameworks or launch GPU-serving processes.

## Workflow

1. Use the machine daemon through the default socket `/var/run/gpu-lease.sock`.
   Do not start a new daemon for routine GPU work. Do not pass `--socket` or set
   `GPU_LEASE_SOCKET` unless the user explicitly provides another socket.

2. Wrap every GPU command with `gpu-lease run`. By default, request the number of GPUs
   you need with `--count` and include `--wait` so the command starts when GPUs are
   ready:

   ```bash
   gpu-lease run --count 2 --wait -- python train.py --batch-size 8
   ```

   Use exact GPU IDs only when the user specifically requires fixed devices:

   ```bash
   gpu-lease run --ids 0,1 -- python train.py --batch-size 8
   ```

3. Let `gpu-lease run` own `CUDA_VISIBLE_DEVICES`. Do not set it separately unless you
   are intentionally composing with another scheduler.

4. Keep the GPU process as the direct child of `gpu-lease run`. The lease is released
   when that command exits.

## Examples

```bash
gpu-lease run --count 1 --wait -- python -m torch.distributed.run --nproc_per_node=1 train.py
gpu-lease run --count 2 --wait -- python -m sglang.launch_server --model-path ./model
gpu-lease run --count 4 --wait -- ray start --head --num-gpus=4
```
