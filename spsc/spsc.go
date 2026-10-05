package spsc

import (
	"sync"
	"sync/atomic"
)

const cacheLine = 64

func roundPow2(n int) int {
	c := 1
	for c < n {
		c <<= 1
	}
	return c
}

type MutexRing[T any] struct {
	mu         sync.Mutex
	buf        []T
	mask       uint64
	head, tail uint64
}

func NewMutexRing[T any](capacity int) *MutexRing[T] {
	c := roundPow2(capacity)
	return &MutexRing[T]{buf: make([]T, c), mask: uint64(c - 1)}
}

func (q *MutexRing[T]) Push(v T) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.tail-q.head == uint64(len(q.buf)) {
		return false
	}
	q.buf[q.tail&q.mask] = v
	q.tail++
	return true
}

func (q *MutexRing[T]) Pop() (T, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	var zero T
	if q.tail == q.head {
		return zero, false
	}
	v := q.buf[q.head&q.mask]
	q.head++
	return v, true
}

type NaiveRing[T any] struct {
	head atomic.Uint64
	tail atomic.Uint64
	buf  []T
	mask uint64
}

func NewNaiveRing[T any](capacity int) *NaiveRing[T] {
	c := roundPow2(capacity)
	return &NaiveRing[T]{buf: make([]T, c), mask: uint64(c - 1)}
}

func (q *NaiveRing[T]) Push(v T) bool {
	t := q.tail.Load()
	if t-q.head.Load() == uint64(len(q.buf)) {
		return false
	}
	q.buf[t&q.mask] = v
	q.tail.Store(t + 1)
	return true
}

func (q *NaiveRing[T]) Pop() (T, bool) {
	var zero T
	h := q.head.Load()
	if h == q.tail.Load() {
		return zero, false
	}
	v := q.buf[h&q.mask]
	q.buf[h&q.mask] = zero
	q.head.Store(h + 1)
	return v, true
}

type PaddedRing[T any] struct {
	_    [cacheLine]byte
	head atomic.Uint64
	_    [cacheLine - 8]byte
	tail atomic.Uint64
	_    [cacheLine - 8]byte
	buf  []T
	mask uint64
}

func NewPaddedRing[T any](capacity int) *PaddedRing[T] {
	c := roundPow2(capacity)
	return &PaddedRing[T]{buf: make([]T, c), mask: uint64(c - 1)}
}

func (q *PaddedRing[T]) Push(v T) bool {
	t := q.tail.Load()
	if t-q.head.Load() == uint64(len(q.buf)) {
		return false
	}
	q.buf[t&q.mask] = v
	q.tail.Store(t + 1)
	return true
}

func (q *PaddedRing[T]) Pop() (T, bool) {
	var zero T
	h := q.head.Load()
	if h == q.tail.Load() {
		return zero, false
	}
	v := q.buf[h&q.mask]
	q.buf[h&q.mask] = zero
	q.head.Store(h + 1)
	return v, true
}

type Ring[T any] struct {
	_ [cacheLine]byte

	head       atomic.Uint64
	cachedTail uint64
	_          [cacheLine - 16]byte

	tail       atomic.Uint64
	cachedHead uint64
	_          [cacheLine - 16]byte

	buf  []T
	mask uint64
	_    [cacheLine - 32]byte
}

func New[T any](capacity int) *Ring[T] {
	c := roundPow2(capacity)
	return &Ring[T]{buf: make([]T, c), mask: uint64(c - 1)}
}

func (q *Ring[T]) Cap() int { return len(q.buf) }

func (q *Ring[T]) Push(v T) bool {
	t := q.tail.Load()
	if t-q.cachedHead == uint64(len(q.buf)) {
		q.cachedHead = q.head.Load()
		if t-q.cachedHead == uint64(len(q.buf)) {
			return false
		}
	}
	q.buf[t&q.mask] = v
	q.tail.Store(t + 1)
	return true
}

func (q *Ring[T]) Pop() (v T, ok bool) {
	h := q.head.Load()
	if h == q.cachedTail {
		q.cachedTail = q.tail.Load()
		if h == q.cachedTail {
			return v, false
		}
	}
	var zero T
	v = q.buf[h&q.mask]
	q.buf[h&q.mask] = zero
	q.head.Store(h + 1)
	return v, true
}

func (q *Ring[T]) PushBatch(vs []T) int {
	t := q.tail.Load()
	free := uint64(len(q.buf)) - (t - q.cachedHead)
	if free < uint64(len(vs)) {
		q.cachedHead = q.head.Load()
		free = uint64(len(q.buf)) - (t - q.cachedHead)
	}
	n := min(uint64(len(vs)), free)

	start := t & q.mask
	first := copy(q.buf[start:], vs[:n])
	copy(q.buf, vs[first:n])
	if n > 0 {
		q.tail.Store(t + n)
	}
	return int(n)
}

func (q *Ring[T]) PopBatch(dst []T) int {
	h := q.head.Load()
	avail := q.cachedTail - h
	if avail < uint64(len(dst)) {
		q.cachedTail = q.tail.Load()
		avail = q.cachedTail - h
	}
	n := min(uint64(len(dst)), avail)
	start := h & q.mask
	end := min(start+n, uint64(len(q.buf)))
	first := copy(dst[:n], q.buf[start:end])
	clear(q.buf[start:end])
	rest := copy(dst[first:n], q.buf)
	clear(q.buf[:rest])
	if n > 0 {
		q.head.Store(h + n)
	}
	return int(n)
}
