package spsc

import "testing"

func transferSpin(q queue, n uint64) {
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := uint64(1); i <= n; {
			if q.Push(i) {
				i++
			}
		}
	}()
	for got := uint64(0); got < n; {
		if _, ok := q.Pop(); ok {
			got++
		}
	}
	<-done
}

func BenchmarkSpinMutex(b *testing.B)  { transferSpin(NewMutexRing[uint64](benchCap), uint64(b.N)) }
func BenchmarkSpinNaive(b *testing.B)  { transferSpin(NewNaiveRing[uint64](benchCap), uint64(b.N)) }
func BenchmarkSpinPadded(b *testing.B) { transferSpin(NewPaddedRing[uint64](benchCap), uint64(b.N)) }
func BenchmarkSpinCached(b *testing.B) { transferSpin(New[uint64](benchCap), uint64(b.N)) }

func BenchmarkChannelBatch64(b *testing.B) {
	ch := make(chan [64]uint64, benchCap/64)
	n := b.N
	b.ResetTimer()
	go func() {
		var blk [64]uint64
		for sent := 0; sent < n; sent += 64 {
			ch <- blk
		}
	}()
	for got := 0; got < n; got += 64 {
		<-ch
	}
}
