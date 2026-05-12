package lease

import (
	"encoding/json"
	"errors"
	"net"
)

type HeldLease struct {
	ID   string
	conn net.Conn
}

func Acquire(socketPath string, ids []int) (*HeldLease, error) {
	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		return nil, err
	}

	if err := json.NewEncoder(conn).Encode(Request{Action: actionLease, IDs: ids}); err != nil {
		conn.Close()
		return nil, err
	}

	var resp Response
	if err := json.NewDecoder(conn).Decode(&resp); err != nil {
		conn.Close()
		return nil, err
	}
	if !resp.OK {
		conn.Close()
		return nil, errors.New(resp.Error)
	}
	return &HeldLease{ID: resp.Lease, conn: conn}, nil
}

func (l *HeldLease) Close() error {
	if l == nil || l.conn == nil {
		return nil
	}
	return l.conn.Close()
}

func Status(socketPath string) (map[string]string, error) {
	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		return nil, err
	}
	defer conn.Close()

	if err := json.NewEncoder(conn).Encode(Request{Action: actionStatus}); err != nil {
		return nil, err
	}

	var resp Response
	if err := json.NewDecoder(conn).Decode(&resp); err != nil {
		return nil, err
	}
	if !resp.OK {
		return nil, errors.New(resp.Error)
	}
	return resp.Leases, nil
}
