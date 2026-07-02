---
name: gpu-lease
description: Use before running local GPU workloads such as PyTorch, SGLang serving, Ray clusters, CUDA benchmarks, or scripts that use GPUs; instructs agents to wrap commands with gpu-lease run so CUDA_VISIBLE_DEVICES is set through a lease.
---

# GPU Lease

Use this skill before running local GPU workloads from Codex or another code agent.

GPU workloads include PyTorch training or inference, SGLang serving, Ray workers or clusters,
CUDA benchmarks, and scripts that import GPU frameworks or launch GPU-serving processes.

## Workflow

1. Before running GPU work, make sure both the `gpu-lease` binary and the
   default socket exist:

   ```bash
   command -v gpu-lease
   test -S /var/run/gpu-lease.sock
   ```

   If either check fails, automatically try to install and deploy `gpu-lease`
   before falling back to an unleased GPU command. Prefer the existing local
   checkout when present; otherwise clone the source from
   `https://github.com/reyoung/gpu-lease.git`.

   ```bash
   src=/home/josephyu/gpu-lease
   if [ ! -d "$src/.git" ]; then
     src=/tmp/gpu-lease
     if [ ! -d "$src/.git" ]; then
       git clone https://github.com/reyoung/gpu-lease.git "$src"
     fi
   fi
   cmake -S "$src" -B "$src/build"
   cmake --build "$src/build" -j
   sudo install -m 0755 "$src/build/gpu-lease" /usr/bin/gpu-lease
   ```

   Then start the daemon if `gpu-lease status` still cannot reach the socket:

   ```bash
   if ! /usr/bin/gpu-lease status >/dev/null 2>&1; then
     sudo rm -f /var/run/gpu-lease.sock /run/gpu-lease.sock
     sudo sh -c 'nohup /usr/bin/gpu-lease daemon --busy-matmul >>/var/log/gpu-lease.log 2>&1 & echo $!'
   fi
   ```

   Verify the deployment before continuing:

   ```bash
   /usr/bin/gpu-lease status
   stat -c '%A %a %U %G %n' /var/run/gpu-lease.sock /run/gpu-lease.sock
   /usr/bin/gpu-lease run --count 1 --wait -- /bin/sh -c 'echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES'
   ```

2. Use the machine daemon through the default socket `/var/run/gpu-lease.sock`.
   Do not start a new daemon for routine GPU work. Do not pass `--socket` or set
   `GPU_LEASE_SOCKET` unless the user explicitly provides another socket.

3. Wrap every GPU command with `gpu-lease run`. By default, request the number of GPUs
   you need with `--count` and include `--wait` so the command starts when GPUs are
   ready:

   ```bash
   gpu-lease run --count 2 --wait -- python train.py --batch-size 8
   ```

   Use exact GPU IDs only when the user specifically requires fixed devices:

   ```bash
   gpu-lease run --ids 0,1 -- python train.py --batch-size 8
   ```

4. Let `gpu-lease run` own `CUDA_VISIBLE_DEVICES`. Do not set it separately unless you
   are intentionally composing with another scheduler.

5. Keep the GPU process as the direct child of `gpu-lease run`. The lease is released
   when that command exits.

6. The daemon performs cleanup before granting each lease. It kills unmanaged
   compute processes on the leased physical GPUs, then waits for NVML GPU
   utilization to reach zero and for unmanaged compute processes to disappear.
   NVML memory with no compute process may be driver/context residue. Do not
   bypass this unless you are running tests or explicitly debugging lease startup;
   `GPU_LEASE_DISABLE_PRESTART_CHECK=1` disables it in the daemon environment.

## Examples

```bash
gpu-lease run --count 1 --wait -- python -m torch.distributed.run --nproc_per_node=1 train.py
gpu-lease run --count 2 --wait -- python -m sglang.launch_server --model-path ./model
gpu-lease run --count 4 --wait -- ray start --head --num-gpus=4
```
