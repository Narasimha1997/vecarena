//go:build !linux && !darwin

package vecarena

import (
	"errors"
	"unsafe"
)

// mapRegion falls back to a Go heap allocation where mmap is unavailable.
// A []byte holds no pointers, so the garbage collector never scans it.
// The slice is re-sliced to start on a 64-byte boundary. HugePages is ignored.
func mapRegion(size int, opt Options) ([]byte, error) {
	if opt.Lock {
		return nil, errors.New("vecarena: Lock is not supported on this platform")
	}
	buf := make([]byte, size+cacheLine)
	off := (cacheLine - int(uintptr(unsafe.Pointer(&buf[0]))%cacheLine)) % cacheLine
	return buf[off : off+size : off+size], nil
}

func unmapRegion([]byte) error { return nil }
