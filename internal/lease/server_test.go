package lease

import (
	"context"
	"path/filepath"
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
