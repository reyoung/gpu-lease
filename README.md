# gpu-lease

`gpu-lease` leases explicit GPU IDs through a local Unix socket daemon and runs a child
process with `CUDA_VISIBLE_DEVICES` set to the leased IDs. The lease is held for the
lifetime of the client connection and is released when the child process exits.

## Usage

Start the daemon:

```bash
gpu-lease daemon /var/run/gpu-lease.sock
```

Run a GPU command:

```bash
gpu-lease run --ids 0,1,2,3 -- some_command --with -args
```

Inspect active leases:

```bash
gpu-lease status
```

The default socket path is `/var/run/gpu-lease.sock`. Set `GPU_LEASE_SOCKET` or pass
`--socket PATH` to use a different socket.

The daemon command also accepts the misspelled alias `deamon` for compatibility with
older examples.

## Development

```bash
go test ./...
go build ./cmd/gpu-lease
```
