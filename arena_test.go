//go:build linux

package vecarena

import (
	"math/rand"
	"sync"
	"testing"
	"unsafe"
)

func TestAddGetDeleteReuse(t *testing.T) {
	a, err := New(3, 1000, Options{HugePages: true})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	if a.Stride() != 16 {
		t.Fatalf("stride = %d, want 16", a.Stride())
	}
	id0, _ := a.Add([]float32{1, 2, 3})
	id1, _ := a.Add([]float32{4, 5, 6})
	if g := a.Get(id1); g[0] != 4 || g[2] != 6 {
		t.Fatalf("got %v", g)
	}
	if uintptr(unsafe.Pointer(&a.Get(id1)[0]))%64 != 0 {
		t.Fatal("row not cache-line aligned")
	}
	a.Delete(id0)
	var seen []uint32
	a.Range(0, 1000, func(id uint32, _ []float32) { seen = append(seen, id) })
	if len(seen) != 1 || seen[0] != id1 {
		t.Fatalf("range saw %v", seen)
	}
	id2, _ := a.Add([]float32{7, 8, 9})
	if id2 != id0 || a.Len() != 2 {
		t.Fatalf("slot not reused: id2=%d len=%d", id2, a.Len())
	}
}

func TestRangeSkipsWholeWords(t *testing.T) {
	a, _ := New(4, 1000, Options{})
	defer a.Close()
	for i := 0; i < 300; i++ {
		a.Add([]float32{float32(i), 0, 0, 0})
	}
	for i := 0; i < 200; i++ {
		a.Delete(uint32(i))
	}
	n := 0
	a.Range(0, 1000, func(id uint32, v []float32) {
		if int(v[0]) != int(id) || id < 200 {
			t.Fatalf("bad row %d", id)
		}
		n++
	})
	if n != 100 {
		t.Fatalf("n = %d", n)
	}
}

func TestConcurrentReadersOneWriter(t *testing.T) {
	a, _ := New(8, 100000, Options{})
	defer a.Close()
	var wg sync.WaitGroup
	stop := make(chan struct{})
	for r := 0; r < 4; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-stop:
					return
				default:
				}
				a.Range(0, 100000, func(id uint32, v []float32) {
					if v[7] != 7 {
						t.Errorf("torn row %d", id)
					}
				})
			}
		}()
	}
	v := []float32{0, 1, 2, 3, 4, 5, 6, 7}
	for i := 0; i < 50000; i++ {
		a.Add(v)
		if i%7 == 0 {
			a.Delete(uint32(i / 2))
		}
	}
	close(stop)
	wg.Wait()
}

func dot(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return s0 + s1 + s2 + s3
}

func BenchmarkScan100k768(b *testing.B) {
	const n, d = 100_000, 768
	a, _ := New(d, n, Options{HugePages: true})
	defer a.Close()
	v := make([]float32, d)
	for i := 0; i < n; i++ {
		for j := range v {
			v[j] = rand.Float32()
		}
		a.Add(v)
	}
	q := v
	b.SetBytes(int64(n * d * 4))
	b.ResetTimer()
	for it := 0; it < b.N; it++ {
		var best float32
		a.Range(0, n, func(_ uint32, row []float32) {
			if s := dot(q, row); s > best {
				best = s
			}
		})
	}
}
