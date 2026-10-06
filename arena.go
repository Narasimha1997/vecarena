package vecarena

import (
	"errors"
	"math/bits"
	"sync"
	"sync/atomic"
	"unsafe"

	"vecarena/kernel"
)

const cacheLine = 64

type Arena struct {
	dim       int
	stride    int
	maxRows   int
	normalize bool
	mem       []byte
	data      []float32

	count atomic.Int64
	live  atomic.Int64

	wmu     sync.Mutex
	dead    []uint64
	free    []uint32
	pending []retired

	// Epoch-based reclamation (B-01): a deleted id goes to pending, tagged
	// with the epoch it was deleted in, and moves to free only once no reader
	// that could still be reading it remains. Readers count themselves in
	// readers[epoch&1]; only epochs E and E-1 can have readers at any time.
	epoch   atomic.Uint64
	readers [2]paddedCounter
}

type paddedCounter struct {
	n atomic.Int64
	_ [cacheLine - 8]byte
}

type retired struct {
	id    uint32
	epoch uint64
}

// Guard marks a reader as active. Views from Get and Block, and ids seen by
// Range, stay valid (are not reused for new rows) until Exit.
type Guard struct {
	a *Arena
	e uint64
}

type Options struct {
	HugePages bool // madvise(MADV_HUGEPAGE) on Linux; ignored elsewhere
	Lock      bool // mlock the whole region; fails if the platform cannot
	Normalize bool // store every added vector scaled to unit length (cosine similarity)
}

func New(dim, maxRows int, opt Options) (*Arena, error) {
	if dim <= 0 || maxRows <= 0 {
		return nil, errors.New("vecarena: dim and maxRows must be > 0")
	}
	per := cacheLine / 4
	stride := (dim + per - 1) / per * per
	size := stride * maxRows * 4

	mem, err := mapRegion(size, opt)
	if err != nil {
		return nil, err
	}
	a := &Arena{
		dim: dim, stride: stride, maxRows: maxRows, mem: mem, normalize: opt.Normalize,
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

	if len(a.free) == 0 {
		a.reclaim()
	}
	var id uint32
	if n := len(a.free); n > 0 {
		id = a.free[n-1]
		a.free = a.free[:n-1]
		a.write(id, v)
		atomic.AndUint64(&a.dead[id/64], ^(1 << (id % 64)))
	} else {
		c := a.count.Load()
		if int(c) >= a.maxRows {
			return 0, errors.New("vecarena: full")
		}
		id = uint32(c)
		a.write(id, v)
		a.count.Store(c + 1)
	}
	a.live.Add(1)
	return id, nil
}

// write fills row id from v before the row is published. The row is
// normalized in place, so the caller's slice is never modified.
func (a *Arena) write(id uint32, v []float32) {
	row := a.row(id)[:a.dim]
	copy(row, v)
	if a.normalize {
		kernel.Normalize(row)
	}
}

func (a *Arena) Delete(id uint32) {
	a.wmu.Lock()
	defer a.wmu.Unlock()
	if int64(id) >= a.count.Load() || a.isDead(id) {
		return
	}
	atomic.OrUint64(&a.dead[id/64], 1<<(id%64))
	// Readers that enter after this epoch see the dead bit, so only readers
	// from this epoch or earlier can still be reading the row.
	a.pending = append(a.pending, retired{id, a.epoch.Load()})
	a.live.Add(-1)
	a.reclaim()
}

// Enter registers the caller as a reader. Pair every Enter with Exit, and
// keep the critical section short: deleted ids are not reused while it lasts.
func (a *Arena) Enter() Guard {
	for {
		e := a.epoch.Load()
		c := &a.readers[e&1].n
		c.Add(1)
		// If the epoch moved before the increment was visible, the writer may
		// already have checked this counter, so retry in the new epoch.
		if a.epoch.Load() == e {
			return Guard{a, e}
		}
		c.Add(-1)
	}
}

// Exit ends the read section started by Enter.
func (g Guard) Exit() { g.a.readers[g.e&1].n.Add(-1) }

// reclaim advances the epoch while no reader from the previous epoch remains,
// and moves ids deleted at least two epochs ago to free. Caller holds wmu.
func (a *Arena) reclaim() {
	for len(a.pending) > 0 {
		e := a.epoch.Load()
		i := 0
		for i < len(a.pending) && a.pending[i].epoch+2 <= e {
			a.free = append(a.free, a.pending[i].id)
			i++
		}
		if i > 0 {
			a.pending = append(a.pending[:0], a.pending[i:]...)
			continue
		}
		// readers[(e+1)&1] counts readers of epoch e-1. Once they are gone,
		// every reader is in epoch e or later and the epoch can advance.
		if a.readers[(e+1)&1].n.Load() != 0 {
			return
		}
		a.epoch.Store(e + 1)
	}
}

func (a *Arena) Get(id uint32) []float32 { return a.row(id)[:a.dim] }

func (a *Arena) row(id uint32) []float32 {
	off := int(id) * a.stride
	return a.data[off : off+a.stride : off+a.stride]
}

// IsLive reports whether id holds a row that has not been deleted. It is safe
// to call concurrently with writers and can be passed as search.Index.Live.
func (a *Arena) IsLive(id uint32) bool {
	return int64(id) < a.count.Load() && !a.isDead(id)
}

func (a *Arena) isDead(id uint32) bool {
	return atomic.LoadUint64(&a.dead[id/64])&(1<<(id%64)) != 0
}

func (a *Arena) Dim() int    { return a.dim }
func (a *Arena) Stride() int { return a.stride }
func (a *Arena) Len() int    { return int(a.live.Load()) }

func (a *Arena) Block(lo, hi int) []float32 { return a.data[lo*a.stride : hi*a.stride] }

func (a *Arena) Range(lo, hi int, fn func(id uint32, v []float32)) {
	g := a.Enter()
	defer g.Exit()
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
	return unmapRegion(a.mem)
}
