package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"sort"
	"strings"
	"syscall"

	"github.com/reyoung/gpu-lease/internal/lease"
)

func main() {
	os.Exit(run(os.Args[1:]))
}

func run(args []string) int {
	if len(args) == 0 {
		usage(os.Stderr)
		return 2
	}

	switch args[0] {
	case "daemon", "deamon":
		return daemon(args[1:])
	case "run":
		return runCommand(args[1:])
	case "status":
		return status(args[1:])
	case "-h", "--help", "help":
		usage(os.Stdout)
		return 0
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n", args[0])
		usage(os.Stderr)
		return 2
	}
}

func usage(out *os.File) {
	fmt.Fprintf(out, `Usage:
  gpu-lease daemon [--socket PATH] [PATH]
  gpu-lease run --ids 0,1 -- command [args...]
  gpu-lease status [--socket PATH]

Environment:
  GPU_LEASE_SOCKET overrides the default socket path (%s).
`, lease.DefaultSocketPath)
}

func daemon(args []string) int {
	fs := flag.NewFlagSet("daemon", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	socketFlag := fs.String("socket", "", "unix socket path")
	if err := fs.Parse(args); err != nil {
		return 2
	}

	socketPath := lease.SocketPath(*socketFlag)
	if fs.NArg() > 1 {
		fmt.Fprintln(os.Stderr, "daemon accepts at most one socket path")
		return 2
	}
	if fs.NArg() == 1 {
		socketPath = fs.Arg(0)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := lease.NewServer(nil).ListenAndServe(ctx, socketPath); err != nil {
		fmt.Fprintf(os.Stderr, "daemon: %v\n", err)
		return 1
	}
	return 0
}

func runCommand(args []string) int {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	idsRaw := fs.String("ids", "", "comma-separated GPU IDs to lease")
	socketFlag := fs.String("socket", "", "unix socket path")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "run requires a command after --")
		return 2
	}

	ids, err := lease.ParseIDs(*idsRaw)
	if err != nil {
		fmt.Fprintf(os.Stderr, "run: %v\n", err)
		return 2
	}

	held, err := lease.Acquire(lease.SocketPath(*socketFlag), ids)
	if err != nil {
		fmt.Fprintf(os.Stderr, "run: acquire lease: %v\n", err)
		return 1
	}
	defer held.Close()

	cmdArgs := fs.Args()
	cmd := exec.Command(cmdArgs[0], cmdArgs[1:]...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = appendEnv(os.Environ(), "CUDA_VISIBLE_DEVICES", lease.IDsEnv(ids))

	if err := cmd.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			return exitErr.ExitCode()
		}
		fmt.Fprintf(os.Stderr, "run: command failed: %v\n", err)
		return 1
	}
	return 0
}

func appendEnv(env []string, key, value string) []string {
	prefix := key + "="
	out := make([]string, 0, len(env)+1)
	for _, item := range env {
		if strings.HasPrefix(item, prefix) {
			continue
		}
		out = append(out, item)
	}
	return append(out, prefix+value)
}

func status(args []string) int {
	fs := flag.NewFlagSet("status", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	socketFlag := fs.String("socket", "", "unix socket path")
	if err := fs.Parse(args); err != nil {
		return 2
	}

	leases, err := lease.Status(lease.SocketPath(*socketFlag))
	if err != nil {
		fmt.Fprintf(os.Stderr, "status: %v\n", err)
		return 1
	}

	ids := make([]string, 0, len(leases))
	for id := range leases {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		fmt.Printf("%s %s\n", id, leases[id])
	}
	return 0
}
