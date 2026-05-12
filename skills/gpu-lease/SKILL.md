# GPU Lease

Use this skill before running local GPU workloads from Codex or another code agent.

GPU workloads include PyTorch training or inference, SGLang serving, Ray workers or clusters,
CUDA benchmarks, and scripts that import GPU frameworks or launch GPU-serving processes.

## Workflow

1. Start or locate the daemon for the machine:

   ```bash
   gpu-lease daemon /var/run/gpu-lease.sock
   ```

   The default socket is `/var/run/gpu-lease.sock`. Override it with `GPU_LEASE_SOCKET`
   or pass `--socket PATH` to `gpu-lease` commands.

2. Wrap every GPU command with `gpu-lease run` and the exact GPU IDs you intend to use:

   ```bash
   gpu-lease run --ids 0,1 -- python train.py --batch-size 8
   ```

3. Let `gpu-lease run` own `CUDA_VISIBLE_DEVICES`. Do not set it separately unless you
   are intentionally composing with another scheduler.

4. Keep the GPU process as the direct child of `gpu-lease run`. The lease is released
   when that command exits.

## Examples

```bash
gpu-lease run --ids 0 -- python -m torch.distributed.run --nproc_per_node=1 train.py
gpu-lease run --ids 2,3 -- python -m sglang.launch_server --model-path ./model
gpu-lease run --ids 0,1,2,3 -- ray start --head --num-gpus=4
```
