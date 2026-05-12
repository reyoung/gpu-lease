package lease

import (
	"encoding/json"
	"errors"
	"net"
)

type HeldLease struct {
	ID   string
	IDs  []int
	conn net.Conn
}

type AcquireOptions struct {
	IDs   []int
	Count int
	Wait  bool
}

func Acquire(socketPath string, ids []int) (*HeldLease, error) {
	return AcquireWithOptions(socketPath, AcquireOptions{IDs: ids})
}

func AcquireWithOptions(socketPath string, opts AcquireOptions) (*HeldLease, error) {
	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		return nil, err
	}

	req := Request{
		Action: actionLease,
		IDs:    opts.IDs,
		Count:  opts.Count,
		Wait:   opts.Wait,
	}
	if err := json.NewEncoder(conn).Encode(req); err != nil {
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
	return &HeldLease{ID: resp.Lease, IDs: resp.IDs, conn: conn}, nil
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
