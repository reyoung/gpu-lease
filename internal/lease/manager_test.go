package lease

import (
	"testing"
	"time"
)

func TestManagerLeaseConflictAndRelease(t *testing.T) {
	manager := NewManager()

	first, err := manager.Lease([]int{0, 1})
	if err != nil {
		t.Fatalf("first lease failed: %v", err)
	}
	if _, err := manager.Lease([]int{1, 2}); err == nil {
		t.Fatal("conflicting lease succeeded")
	}

	status := manager.Status()
	if status["0"] != first || status["1"] != first {
		t.Fatalf("status = %#v, want ids 0 and 1 held by %q", status, first)
	}

	manager.Release(first)
	if got := manager.Status(); len(got) != 0 {
		t.Fatalf("status after release = %#v, want empty", got)
	}

	if _, err := manager.Lease([]int{1, 2}); err != nil {
		t.Fatalf("lease after release failed: %v", err)
	}
}

func TestManagerLeaseAny(t *testing.T) {
	manager := NewManagerWithAvailableIDs([]int{2, 0, 1})

	leaseID, ids, err := manager.LeaseAny(2, false)
	if err != nil {
		t.Fatalf("LeaseAny failed: %v", err)
	}
	if got := IDsEnv(ids); got != "0,1" {
		t.Fatalf("LeaseAny ids = %q, want 0,1", got)
	}

	_, ids, err = manager.LeaseAny(2, false)
	if err == nil {
		t.Fatalf("conflicting LeaseAny succeeded with ids %#v", ids)
	}

	manager.Release(leaseID)
}

func TestManagerLeaseAnyWaits(t *testing.T) {
	manager := NewManagerWithAvailableIDs([]int{0})

	leaseID, _, err := manager.LeaseAny(1, false)
	if err != nil {
		t.Fatalf("initial LeaseAny failed: %v", err)
	}

	done := make(chan []int, 1)
	errCh := make(chan error, 1)
	go func() {
		_, ids, err := manager.LeaseAny(1, true)
		if err != nil {
			errCh <- err
			return
		}
		done <- ids
	}()

	select {
	case ids := <-done:
		t.Fatalf("LeaseAny returned before release with ids %#v", ids)
	case err := <-errCh:
		t.Fatalf("LeaseAny returned error before release: %v", err)
	case <-time.After(50 * time.Millisecond):
	}

	manager.Release(leaseID)

	select {
	case ids := <-done:
		if got := IDsEnv(ids); got != "0" {
			t.Fatalf("LeaseAny ids = %q, want 0", got)
		}
	case err := <-errCh:
		t.Fatalf("LeaseAny returned error: %v", err)
	case <-time.After(2 * time.Second):
		t.Fatal("LeaseAny did not return after release")
	}
}
