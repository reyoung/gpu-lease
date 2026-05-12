package main

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/reyoung/gpu-lease/internal/lease"
)

func TestAppendEnvReplacesExistingValue(t *testing.T) {
	env := appendEnv([]string{"A=1", "CUDA_VISIBLE_DEVICES=9"}, "CUDA_VISIBLE_DEVICES", "0,1")
	want := map[string]bool{"A=1": true, "CUDA_VISIBLE_DEVICES=0,1": true}
	for _, item := range env {
		if !want[item] {
			t.Fatalf("unexpected env item %q in %#v", item, env)
		}
		delete(want, item)
	}
	if len(want) != 0 {
		t.Fatalf("missing env items: %#v", want)
	}
}

func TestRunCommandSetsCUDAVisibleDevicesAndReleasesLease(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix socket command test")
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	socketPath := filepath.Join(t.TempDir(), "gpu-lease.sock")
	errCh := make(chan error, 1)
	go func() {
		errCh <- lease.NewServer(nil).ListenAndServe(ctx, socketPath)
	}()
	waitForSocket(t, socketPath)

	code := run([]string{
		"run",
		"--socket", socketPath,
		"--ids", "0,1",
		"--",
		"/bin/sh", "-c", `test "$CUDA_VISIBLE_DEVICES" = "0,1"`,
	})
	if code != 0 {
		t.Fatalf("run returned %d, want 0", code)
	}

	deadline := time.Now().Add(2 * time.Second)
	for {
		status, err := lease.Status(socketPath)
		if err != nil {
			t.Fatalf("Status failed: %v", err)
		}
		if len(status) == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("lease was not released after command exit: %#v", status)
		}
		time.Sleep(10 * time.Millisecond)
	}

	cancel()
	if err := <-errCh; err != nil {
		t.Fatalf("server returned error: %v", err)
	}
}

func TestRunCommandCanLeaseByCount(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix socket command test")
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	socketPath := filepath.Join(t.TempDir(), "gpu-lease.sock")
	errCh := make(chan error, 1)
	go func() {
		errCh <- lease.NewServer(lease.NewManagerWithAvailableIDs([]int{0, 1, 2})).ListenAndServe(ctx, socketPath)
	}()
	waitForSocket(t, socketPath)

	code := run([]string{
		"run",
		"--socket", socketPath,
		"--num", "2",
		"--",
		"/bin/sh", "-c", `test "$CUDA_VISIBLE_DEVICES" = "0,1"`,
	})
	if code != 0 {
		t.Fatalf("run returned %d, want 0", code)
	}

	cancel()
	if err := <-errCh; err != nil {
		t.Fatalf("server returned error: %v", err)
	}
}

func TestRunCommandWaitsForCount(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("unix socket command test")
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	socketPath := filepath.Join(t.TempDir(), "gpu-lease.sock")
	errCh := make(chan error, 1)
	go func() {
		errCh <- lease.NewServer(lease.NewManagerWithAvailableIDs([]int{0})).ListenAndServe(ctx, socketPath)
	}()
	waitForSocket(t, socketPath)

	held, err := lease.AcquireWithOptions(socketPath, lease.AcquireOptions{Count: 1})
	if err != nil {
		t.Fatalf("AcquireWithOptions failed: %v", err)
	}

	done := make(chan int, 1)
	go func() {
		done <- run([]string{
			"run",
			"--socket", socketPath,
			"--num", "1",
			"--wait",
			"--",
			"/bin/sh", "-c", `test "$CUDA_VISIBLE_DEVICES" = "0"`,
		})
	}()

	select {
	case code := <-done:
		t.Fatalf("run returned %d before lease was released", code)
	case <-time.After(50 * time.Millisecond):
	}

	if err := held.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	select {
	case code := <-done:
		if code != 0 {
			t.Fatalf("run returned %d, want 0", code)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("run did not return after lease was released")
	}

	cancel()
	if err := <-errCh; err != nil {
		t.Fatalf("server returned error: %v", err)
	}
}

func waitForSocket(t *testing.T, socketPath string) {
	t.Helper()

	deadline := time.Now().Add(2 * time.Second)
	for {
		held, err := lease.Acquire(socketPath, []int{99})
		if err == nil {
			_ = held.Close()
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("socket %s did not become ready: %v", socketPath, err)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestMain(m *testing.M) {
	os.Exit(m.Run())
}
