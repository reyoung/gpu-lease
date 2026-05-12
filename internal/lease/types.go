package lease

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
)

const (
	DefaultSocketPath = "/var/run/gpu-lease.sock"

	actionLease  = "lease"
	actionStatus = "status"
)

type Request struct {
	Action string `json:"action"`
	IDs    []int  `json:"ids,omitempty"`
	Count  int    `json:"count,omitempty"`
	Wait   bool   `json:"wait,omitempty"`
}

type Response struct {
	OK     bool              `json:"ok"`
	Error  string            `json:"error,omitempty"`
	Lease  string            `json:"lease,omitempty"`
	IDs    []int             `json:"ids,omitempty"`
	Leases map[string]string `json:"leases,omitempty"`
}

func SocketPath(flagValue string) string {
	if flagValue != "" {
		return flagValue
	}
	if env := os.Getenv("GPU_LEASE_SOCKET"); env != "" {
		return env
	}
	return DefaultSocketPath
}

func ParseIDs(raw string) ([]int, error) {
	if strings.TrimSpace(raw) == "" {
		return nil, errors.New("ids must not be empty")
	}

	parts := strings.Split(raw, ",")
	ids := make([]int, 0, len(parts))
	seen := map[int]struct{}{}
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			return nil, fmt.Errorf("invalid empty GPU id in %q", raw)
		}
		id, err := strconv.Atoi(part)
		if err != nil || id < 0 {
			return nil, fmt.Errorf("invalid GPU id %q", part)
		}
		if _, ok := seen[id]; ok {
			return nil, fmt.Errorf("duplicate GPU id %d", id)
		}
		seen[id] = struct{}{}
		ids = append(ids, id)
	}
	return ids, nil
}

func IDsEnv(ids []int) string {
	parts := make([]string, len(ids))
	for i, id := range ids {
		parts[i] = strconv.Itoa(id)
	}
	return strings.Join(parts, ",")
}
