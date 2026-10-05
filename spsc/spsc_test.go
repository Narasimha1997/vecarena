package spsc

import (
	"runtime"
	"testing"
	"unsafe"
)

type queue interface {
	Push(uint64) bool
	Pop() (uint64, bool)
}

func transfer(tb testing.TB, q queue, n uint64) {
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := uint64(1); i <= n; {
			if q.Push(i) {
				i++
			} else {
				runtime.Gosched()
			}
		}
	}()
	for want := uint64(1); want <= n; {
		v, ok := q.Pop()
		if !ok {
			runtime.Gosched()
			continue
		}
		if v != want {
			tb.Fatalf("got %d, want %d", v, want)
		}
		want++
	}
	<-done
}

func TestFIFOAllVariants(t *testing.T) {
	for name, q := range map[string]queue{
		"mutex":  NewMutexRing[uint64](64),
		"naive":  NewNaiveRing[uint64](64),
		"padded": NewPaddedRing[uint64](64),
		"cached": New[uint64](64),
	} {
		t.Run(name, func(t *testing.T) { transfer(t, q, 200_000) })
	}
}

func TestFullAndEmpty(t *testing.T) {
	q := New[int](4)
	if _, ok := q.Pop(); ok {
		t.Fatal("pop from empty succeeded")
	}
	for i := 0; i < 4; i++ {
		if !q.Push(i) {
			t.Fatalf("push %d failed", i)
		}
	}
	if q.Push(99) {
		t.Fatal("push into full queue succeeded")
	}
	for i := 0; i < 4; i++ {
		if v, ok := q.Pop(); !ok || v != i {
			t.Fatalf("pop = %d,%v want %d", v, ok, i)
		}
	}

	for round := 0; round < 1000; round++ {
		q.Push(round)
		if v, _ := q.Pop(); v != round {
			t.Fatal("wraparound broke order")
		}
	}
}

func TestBatchOrder(t *testing.T) {
	q := New[uint64](256)
	const n = 300_000
	done := make(chan struct{})
	go func() {
		defer close(done)
		buf := make([]uint64, 37)
		next := uint64(1)
		for next <= n {
			k := 0
			for k < len(buf) && next+uint64(k) <= n {
				buf[k] = next + uint64(k)
				k++
			}
			pushed := q.PushBatch(buf[:k])
			next += uint64(pushed)
			if pushed == 0 {
				runtime.Gosched()
			}
		}
	}()
	dst := make([]uint64, 64)
	for want := uint64(1); want <= n; {
		got := q.PopBatch(dst)
		if got == 0 {
			runtime.Gosched()
			continue
		}
		for _, v := range dst[:got] {
			if v != want {
				t.Fatalf("got %d want %d", v, want)
			}
			want++
		}
	}
	<-done
}

func TestLayout(t *testing.T) {
	var r Ring[uint64]
	h := unsafe.Offsetof(r.head)
	tl := unsafe.Offsetof(r.tail)
	if tl-h < cacheLine {
		t.Fatalf("head at %d and tail at %d share a cache line", h, tl)
	}
	var n NaiveRing[uint64]
	t.Logf("naive: head@%d tail@%d (same line); cached: head@%d tail@%d",
		unsafe.Offsetof(n.head), unsafe.Offsetof(n.tail), h, tl)
}

const benchCap = 1024

func benchQueue(b *testing.B, q queue) {
	b.ReportAllocs()
	b.ResetTimer()
	transfer(b, q, uint64(b.N))
}

func BenchmarkChannel(b *testing.B) {
	ch := make(chan uint64, benchCap)
	b.ResetTimer()
	go func() {
		for i := 0; i < b.N; i++ {
			ch <- uint64(i)
		}
	}()
	for i := 0; i < b.N; i++ {
		<-ch
	}
}

func BenchmarkMutex(b *testing.B)  { benchQueue(b, NewMutexRing[uint64](benchCap)) }
func BenchmarkNaive(b *testing.B)  { benchQueue(b, NewNaiveRing[uint64](benchCap)) }
func BenchmarkPadded(b *testing.B) { benchQueue(b, NewPaddedRing[uint64](benchCap)) }
func BenchmarkCached(b *testing.B) { benchQueue(b, New[uint64](benchCap)) }

func BenchmarkCachedBatch64(b *testing.B) {
	q := New[uint64](benchCap)
	n := b.N
	b.ResetTimer()
	go func() {
		buf := make([]uint64, 64)
		for sent := 0; sent < n; {
			k := min(len(buf), n-sent)
			p := q.PushBatch(buf[:k])
			sent += p
			if p == 0 {
				runtime.Gosched()
			}
		}
	}()
	dst := make([]uint64, 64)
	for got := 0; got < n; {
		k := q.PopBatch(dst)
		got += k
		if k == 0 {
			runtime.Gosched()
		}
	}
}
