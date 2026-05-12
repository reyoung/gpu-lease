package lease

import "testing"

func TestParseIDs(t *testing.T) {
	ids, err := ParseIDs("0, 2,3")
	if err != nil {
		t.Fatalf("ParseIDs returned error: %v", err)
	}
	want := []int{0, 2, 3}
	if len(ids) != len(want) {
		t.Fatalf("len(ids) = %d, want %d", len(ids), len(want))
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("ids[%d] = %d, want %d", i, ids[i], want[i])
		}
	}

	for _, input := range []string{"", "0,,1", "-1", "abc", "1,1"} {
		if _, err := ParseIDs(input); err == nil {
			t.Fatalf("ParseIDs(%q) succeeded, want error", input)
		}
	}
}

func TestIDsEnv(t *testing.T) {
	if got := IDsEnv([]int{0, 3, 4}); got != "0,3,4" {
		t.Fatalf("IDsEnv returned %q", got)
	}
}
