//go:build linux

package vecarena

import (
	"errors"
	"math/bits"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"
)

const cacheLine = 64

type Arena struct {
	dim     int
	stride  int
	maxRows int
	mem     []byte
	data    []float32

	count atomic.Int64
	live  atomic.Int64

	wmu  sync.Mutex
	dead []uint64
	free []uint32
}

type Options struct {
	HugePages bool
	Lock      bool
}

func New(dim, maxRows int, opt Options) (*Arena, error) {
	if dim <= 0 || maxRows <= 0 {
		return nil, errors.New("vecarena: dim and maxRows must be > 0")
	}
	per := cacheLine / 4
	stride := (dim + per - 1) / per * per
	size := stride * maxRows * 4

	mem, err := syscall.Mmap(-1, 0, size,
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_PRIVATE|syscall.MAP_ANON|syscall.MAP_NORESERVE)
	if err != nil {
		return nil, err
	}
	if opt.HugePages {
		_ = syscall.Madvise(mem, 14)
	}
	if opt.Lock {
		if err := syscall.Mlock(mem); err != nil {
			syscall.Munmap(mem)
			return nil, err
		}
	}
	a := &Arena{
		dim: dim, stride: stride, maxRows: maxRows, mem: mem,
		data: unsafe.Slice((*float32)(unsafe.Pointer(&mem[0])), stride*maxRows),
		dead: make([]uint64, (maxRows+63)/64),
	}
	return a, nil
}

func (a *Arena) Add(v []float32) (uint32, error) {
	if len(v) != a.dim {
		return 0, errors.New("vecarena: wrong dimension")
	}
	a.wmu.Lock()
	defer a.wmu.Unlock()

	var id uint32
	if n := len(a.free); n > 0 {
		id = a.free[n-1]
		a.free = a.free[:n-1]
		copy(a.row(id), v)
		atomic.AndUint64(&a.dead[id/64], ^(1 << (id % 64)))
	} else {
		c := a.count.Load()
		if int(c) >= a.maxRows {
			return 0, errors.New("vecarena: full")
		}
		id = uint32(c)
		copy(a.row(id), v)
		a.count.Store(c + 1)
	}
	a.live.Add(1)
	return id, nil
}

func (a *Arena) Delete(id uint32) {
	a.wmu.Lock()
	defer a.wmu.Unlock()
	if int64(id) >= a.count.Load() || a.isDead(id) {
		return
	}
	atomic.OrUint64(&a.dead[id/64], 1<<(id%64))
	a.free = append(a.free, id)
	a.live.Add(-1)
}

func (a *Arena) Get(id uint32) []float32 { return a.row(id)[:a.dim] }

func (a *Arena) row(id uint32) []float32 {
	off := int(id) * a.stride
	return a.data[off : off+a.stride : off+a.stride]
}

func (a *Arena) isDead(id uint32) bool {
	return atomic.LoadUint64(&a.dead[id/64])&(1<<(id%64)) != 0
}

func (a *Arena) Dim() int    { return a.dim }
func (a *Arena) Stride() int { return a.stride }
func (a *Arena) Len() int    { return int(a.live.Load()) }

func (a *Arena) Block(lo, hi int) []float32 { return a.data[lo*a.stride : hi*a.stride] }

func (a *Arena) Range(lo, hi int, fn func(id uint32, v []float32)) {
	if c := int(a.count.Load()); hi > c {
		hi = c
	}
	for i := lo; i < hi; {
		w := i / 64
		live := ^atomic.LoadUint64(&a.dead[w]) >> (i % 64)
		if live == 0 {
			i = (w + 1) * 64
			continue
		}
		i += bits.TrailingZeros64(live)
		if i >= hi {
			return
		}
		fn(uint32(i), a.row(uint32(i))[:a.dim])
		i++
	}
}

func (a *Arena) Close() error {
	a.data = nil
	return syscall.Munmap(a.mem)
}
