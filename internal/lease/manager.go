package lease

import (
	"fmt"
	"sort"
	"strconv"
	"sync"
)

type Manager struct {
	mu     sync.Mutex
	heldBy map[int]string
	nextID uint64
}

func NewManager() *Manager {
	return &Manager{
		heldBy: make(map[int]string),
	}
}

func (m *Manager) Lease(ids []int) (string, error) {
	if len(ids) == 0 {
		return "", fmt.Errorf("ids must not be empty")
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	for _, id := range ids {
		if holder, ok := m.heldBy[id]; ok {
			return "", fmt.Errorf("GPU %d is already leased by %s", id, holder)
		}
	}

	m.nextID++
	leaseID := "lease-" + strconv.FormatUint(m.nextID, 10)
	for _, id := range ids {
		m.heldBy[id] = leaseID
	}
	return leaseID, nil
}

func (m *Manager) Release(leaseID string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for id, holder := range m.heldBy {
		if holder == leaseID {
			delete(m.heldBy, id)
		}
	}
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
