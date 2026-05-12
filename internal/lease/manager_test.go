package lease

import "testing"

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
