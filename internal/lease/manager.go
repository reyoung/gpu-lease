package lease

import (
	"context"
	"fmt"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type Manager struct {
	mu           sync.Mutex
	cond         *sync.Cond
	heldBy       map[int]string
	availableIDs []int
	nextID       uint64
}

func NewManager() *Manager {
	return NewManagerWithAvailableIDs(nil)
}

func NewManagerWithAvailableIDs(ids []int) *Manager {
	manager := &Manager{
		heldBy: make(map[int]string),
	}
	manager.cond = sync.NewCond(&manager.mu)
	manager.availableIDs = normalizeIDs(ids)
	return manager
}

func (m *Manager) Lease(ids []int) (string, error) {
	return m.LeaseIDs(ids, false)
}

func (m *Manager) LeaseIDs(ids []int, wait bool) (string, error) {
	if len(ids) == 0 {
		return "", fmt.Errorf("ids must not be empty")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	for {
		holderID, holder, ok := m.firstHeld(ids)
		if !ok {
			return m.assignLocked(ids), nil
		}
		if !wait {
			return "", fmt.Errorf("GPU %d is already leased by %s", holderID, holder)
		}
		m.cond.Wait()
	}
}

func (m *Manager) LeaseAny(count int, wait bool) (string, []int, error) {
	if count <= 0 {
		return "", nil, fmt.Errorf("count must be greater than 0")
	}
	if len(m.availableIDs) == 0 {
		return "", nil, fmt.Errorf("no GPU inventory is available")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	for {
		ids := m.availableLocked(count)
		if len(ids) == count {
			return m.assignLocked(ids), ids, nil
		}
		if !wait {
			return "", nil, fmt.Errorf("requested %d GPU(s), only %d available", count, len(ids))
		}
		m.cond.Wait()
	}
}

func (m *Manager) assignLocked(ids []int) string {
	m.nextID++
	leaseID := "lease-" + strconv.FormatUint(m.nextID, 10)
	for _, id := range ids {
		m.heldBy[id] = leaseID
	}
	return leaseID
}

func (m *Manager) firstHeld(ids []int) (int, string, bool) {
	for _, id := range ids {
		if holder, ok := m.heldBy[id]; ok {
			return id, holder, true
		}
	}
	return 0, "", false
}

func (m *Manager) availableLocked(count int) []int {
	ids := make([]int, 0, count)
	for _, id := range m.availableIDs {
		if _, held := m.heldBy[id]; held {
			continue
		}
		ids = append(ids, id)
		if len(ids) == count {
			break
		}
	}
	return ids
}

func (m *Manager) Release(leaseID string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for id, holder := range m.heldBy {
		if holder == leaseID {
			delete(m.heldBy, id)
		}
	}
	m.cond.Broadcast()
}

func (m *Manager) Status() map[string]string {
	m.mu.Lock()
	defer m.mu.Unlock()

	ids := make([]int, 0, len(m.heldBy))
	for id := range m.heldBy {
		ids = append(ids, id)
	}
	sort.Ints(ids)

	out := make(map[string]string, len(ids))
	for _, id := range ids {
		out[strconv.Itoa(id)] = m.heldBy[id]
	}
	return out
}

func DiscoverGPUIDs() []int {
	if ids := discoverGPUIDsWithNvidiaSMI(); len(ids) > 0 {
		return ids
	}
	return discoverGPUIDsFromDev()
}

func discoverGPUIDsWithNvidiaSMI() []int {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=index", "--format=csv,noheader").Output()
	if err != nil {
		return nil
	}

	lines := regexp.MustCompile(`\r?\n`).Split(string(out), -1)
	ids := make([]int, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		id, err := strconv.Atoi(line)
		if err == nil && id >= 0 {
			ids = append(ids, id)
		}
	}
	return normalizeIDs(ids)
}

func discoverGPUIDsFromDev() []int {
	matches, err := filepath.Glob("/dev/nvidia[0-9]*")
	if err != nil {
		return nil
	}

	re := regexp.MustCompile(`^nvidia([0-9]+)$`)
	ids := make([]int, 0, len(matches))
	for _, match := range matches {
		parts := re.FindStringSubmatch(filepath.Base(match))
		if len(parts) != 2 {
			continue
		}
		id, err := strconv.Atoi(parts[1])
		if err == nil {
			ids = append(ids, id)
		}
	}
	return normalizeIDs(ids)
}

func normalizeIDs(ids []int) []int {
	seen := make(map[int]struct{}, len(ids))
	out := make([]int, 0, len(ids))
	for _, id := range ids {
		if id < 0 {
			continue
		}
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	sort.Ints(out)
	return out
}
