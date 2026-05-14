package lease

import (
	"context"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

func TestServerLeaseLifecycle(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	socketPath := filepath.Join(t.TempDir(), "gpu-lease.sock")
	manager := NewManager()
	errCh := make(chan error, 1)
	go func() {
		errCh <- NewServer(manager).ListenAndServe(ctx, socketPath)
	}()
	waitForSocket(t, socketPath)

	held, err := Acquire(socketPath, []int{0})
	if err != nil {
		t.Fatalf("Acquire failed: %v", err)
	}
	if _, err := Acquire(socketPath, []int{0}); err == nil {
		t.Fatal("conflicting Acquire succeeded")
	}

	status, err := Status(socketPath)
	if err != nil {
		t.Fatalf("Status failed: %v", err)
	}
	if status["0"] != held.ID {
		t.Fatalf("status = %#v, want GPU 0 held by %q", status, held.ID)
	}

	if err := held.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for {
		status, err = Status(socketPath)
		if err != nil {
			t.Fatalf("Status after close failed: %v", err)
		}
		if len(status) == 0 {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("lease was not released after client close: %#v", status)
		}
		time.Sleep(10 * time.Millisecond)
	}

	cancel()
	if err := <-errCh; err != nil {
		t.Fatalf("server returned error: %v", err)
	}
}

func TestServerLeaseAnyReturnsIDs(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	socketPath := filepath.Join(t.TempDir(), "gpu-lease.sock")
	manager := NewManagerWithAvailableIDs([]int{0, 1})
	errCh := make(chan error, 1)
	go func() {
		errCh <- NewServer(manager).ListenAndServe(ctx, socketPath)
	}()
	waitForSocket(t, socketPath)

	held, err := AcquireWithOptions(socketPath, AcquireOptions{Count: 2})
	if err != nil {
		t.Fatalf("AcquireWithOptions failed: %v", err)
	}
	if got := IDsEnv(held.IDs); got != "0,1" {
		t.Fatalf("held IDs = %q, want 0,1", got)
	}
	if err := held.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	cancel()
	if err := <-errCh; err != nil {
		t.Fatalf("server returned error: %v", err)
	}
}

func TestServerLeaseWaits(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	socketPath := filepath.Join(t.TempDir(), "gpu-lease.sock")
	manager := NewManagerWithAvailableIDs([]int{0})
	errCh := make(chan error, 1)
	go func() {
		errCh <- NewServer(manager).ListenAndServe(ctx, socketPath)
	}()
	waitForSocket(t, socketPath)

	first, err := AcquireWithOptions(socketPath, AcquireOptions{Count: 1})
	if err != nil {
		t.Fatalf("first AcquireWithOptions failed: %v", err)
	}

	done := make(chan *HeldLease, 1)
	acquireErr := make(chan error, 1)
	go func() {
		held, err := AcquireWithOptions(socketPath, AcquireOptions{Count: 1, Wait: true})
		if err != nil {
			acquireErr <- err
			return
		}
		done <- held
	}()

	select {
	case held := <-done:
		_ = held.Close()
		t.Fatal("waiting acquire returned before release")
	case err := <-acquireErr:
		t.Fatalf("waiting acquire returned error before release: %v", err)
	case <-time.After(50 * time.Millisecond):
	}

	if err := first.Close(); err != nil {
		t.Fatalf("first Close failed: %v", err)
	}

	select {
	case held := <-done:
		if got := IDsEnv(held.IDs); got != "0" {
			t.Fatalf("held IDs = %q, want 0", got)
		}
		_ = held.Close()
	case err := <-acquireErr:
		t.Fatalf("waiting acquire returned error: %v", err)
	case <-time.After(2 * time.Second):
		t.Fatal("waiting acquire did not return after release")
	}

	cancel()
	if err := <-errCh; err != nil {
		t.Fatalf("server returned error: %v", err)
	}
}

func TestServerSocketIsWorldAccessibleWithRestrictiveUmask(t *testing.T) {
	oldUmask := syscall.Umask(0077)
	t.Cleanup(func() {
		syscall.Umask(oldUmask)
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	socketPath := filepath.Join(t.TempDir(), "gpu-lease.sock")
	errCh := make(chan error, 1)
	go func() {
		errCh <- NewServer(NewManagerWithAvailableIDs([]int{0})).ListenAndServe(ctx, socketPath)
	}()
	waitForSocket(t, socketPath)

	info, err := os.Stat(socketPath)
	if err != nil {
		t.Fatalf("Stat failed: %v", err)
	}
	if got := info.Mode().Perm(); got != 0666 {
		t.Fatalf("socket mode = %o, want 666", got)
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
		held, err := Acquire(socketPath, []int{99})
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
