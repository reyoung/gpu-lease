package lease

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
)

type Server struct {
	manager *Manager
}

func NewServer(manager *Manager) *Server {
	if manager == nil {
		manager = NewManager()
	}
	return &Server{manager: manager}
}

func (s *Server) ListenAndServe(ctx context.Context, socketPath string) error {
	if socketPath == "" {
		return errors.New("socket path must not be empty")
	}
	if err := os.MkdirAll(filepath.Dir(socketPath), 0755); err != nil {
		return err
	}
	if err := os.RemoveAll(socketPath); err != nil {
		return err
	}

	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		return err
	}
	defer listener.Close()
	defer os.Remove(socketPath)

	go func() {
		<-ctx.Done()
		listener.Close()
	}()

	for {
		conn, err := listener.Accept()
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return err
		}
		go s.handle(conn)
	}
}

func (s *Server) handle(conn net.Conn) {
	defer conn.Close()

	reader := bufio.NewReader(conn)
	encoder := json.NewEncoder(conn)

	var req Request
	if err := json.NewDecoder(reader).Decode(&req); err != nil {
		_ = encoder.Encode(Response{OK: false, Error: fmt.Sprintf("decode request: %v", err)})
		return
	}

	switch req.Action {
	case actionLease:
		if len(req.IDs) > 0 && req.Count > 0 {
			_ = encoder.Encode(Response{OK: false, Error: "ids and count cannot both be set"})
			return
		}

		var (
			leaseID string
			ids     []int
			err     error
		)
		if req.Count > 0 {
			leaseID, ids, err = s.manager.LeaseAny(req.Count, req.Wait)
		} else {
			leaseID, err = s.manager.LeaseIDs(req.IDs, req.Wait)
			ids = req.IDs
		}
		if err != nil {
			_ = encoder.Encode(Response{OK: false, Error: err.Error()})
			return
		}
		defer s.manager.Release(leaseID)

		if err := encoder.Encode(Response{OK: true, Lease: leaseID, IDs: ids}); err != nil {
			return
		}
		_, _ = reader.ReadByte()
	case actionStatus:
		_ = encoder.Encode(Response{OK: true, Leases: s.manager.Status()})
	default:
		_ = encoder.Encode(Response{OK: false, Error: "unknown action"})
	}
}
