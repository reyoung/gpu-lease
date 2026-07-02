# gpu-lease

`gpu-lease` leases GPUs through a local Unix socket daemon and runs a child process
with `CUDA_VISIBLE_DEVICES` set to the leased IDs. The lease is held for the lifetime
of the client connection and is released when the child process exits.

## Usage

Start the daemon:

```bash
gpu-lease daemon /var/run/gpu-lease.sock
```

Optionally keep every daemon-discovered idle GPU busy with a cuBLAS matmul
worker until it is leased:

```bash
gpu-lease daemon --busy-matmul /var/run/gpu-lease.sock
```

Run a GPU command:

```bash
gpu-lease run --ids 0,1,2,3 -- some_command --with -args
```

Or ask the daemon for any available GPUs by count:

```bash
gpu-lease run --num 2 -- some_command --with -args
```

Wait until the requested GPUs are available:

```bash
gpu-lease run --num 2 --wait -- some_command --with -args
```

Before the daemon grants each lease, it uses NVML to clean up the leased physical
GPUs. It kills any compute process on those GPUs that is not a `gpu-lease`
process, then waits until NVML reports `gpu_util == 0` and no unmanaged compute
process remains. NVML can keep driver/context memory charged briefly after a
process exits, so memory with no compute process is treated as driver residue
rather than stale dummy GEMM work.

Inspect active leases:

```bash
gpu-lease status
```

The default socket path is `/var/run/gpu-lease.sock`. Set `GPU_LEASE_SOCKET` or pass
`--socket PATH` to use a different socket.

The daemon command also accepts the misspelled alias `deamon` for compatibility with
older examples.

Pre-lease cleanup is enabled by default in the daemon. For tests or emergency
debugging, set `GPU_LEASE_DISABLE_PRESTART_CHECK=1` in the daemon environment to
skip it. The wait loop can be tuned with `GPU_LEASE_PRESTART_TIMEOUT_MS` and
`GPU_LEASE_PRESTART_POLL_MS`.

## Development

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The project builds a single Linux C++17 binary named `gpu-lease`. CUDA Toolkit is
required at build time because the daemon links against CUDA runtime and cuBLAS
for `--busy-matmul`.
