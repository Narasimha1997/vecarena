package search

import (
	"context"
	"fmt"
	"math/rand"
	"sort"
	"sync"
	"testing"
	"time"
)

func randRows(r *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = r.Float32()*2 - 1
	}
	return v
}

func bruteTopK(rows []float32, stride, n int, q []float32, k int) []uint32 {
	type pair struct {
		id uint32
		s  float64
	}
	all := make([]pair, n)
	for i := 0; i < n; i++ {
		var s float64
		for j, x := range q {
			s += float64(x) * float64(rows[i*stride+j])
		}
		all[i] = pair{uint32(i), s}
	}
	sort.Slice(all, func(a, b int) bool { return all[a].s > all[b].s })
	ids := make([]uint32, k)
	for i := range ids {
		ids[i] = all[i].id
	}
	return ids
}

func TestSearchBatchMatchesBruteForce(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	const n, dim, stride, k = 5000, 100, 112, 10
	rows := make([]float32, n*stride)
	for i := 0; i < n; i++ {
		copy(rows[i*stride:], randRows(r, dim))
	}
	ix := New(rows, stride, n)
	ix.TileRows = 333
	for _, nq := range []int{1, 7, 8, 9, 20} {
		qs := make([][]float32, nq)
		for i := range qs {
			qs[i] = randRows(r, dim)
		}
		got := ix.SearchBatch(qs, k)
		for i, q := range qs {
			want := bruteTopK(rows, stride, n, q, k)
			for j := range want {
				if got[i][j].ID != want[j] {
					t.Fatalf("nq=%d q=%d rank %d: got id %d want %d", nq, i, j, got[i][j].ID, want[j])
				}
			}
		}
	}
}

func TestBatcherGroupsConcurrentRequests(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	const n, stride, k = 2000, 64, 5
	rows := randRows(r, n*stride)
	b := NewBatcher(New(rows, stride, n), 16, 5*time.Millisecond)
	defer b.Close()

	var wg sync.WaitGroup
	for i := 0; i < 40; i++ {
		q := randRows(r, stride)
		wg.Add(1)
		go func() {
			defer wg.Done()
			got, err := b.Search(context.Background(), q, k)
			if err != nil {
				t.Error(err)
				return
			}
			want := bruteTopK(rows, stride, n, q, k)
			for j := range want {
				if got[j].ID != want[j] {
					t.Errorf("rank %d: got %d want %d", j, got[j].ID, want[j])
				}
			}
		}()
	}
	wg.Wait()
}

func BenchmarkThroughput(b *testing.B) {
	const n, d, k = 100_000, 768, 10
	r := rand.New(rand.NewSource(3))
	rows := randRows(r, n*d)
	ix := New(rows, d, n)
	pool := make([][]float32, 64)
	for i := range pool {
		pool[i] = randRows(r, d)
	}
	for _, batch := range []int{1, 8, 32, 64} {
		b.Run(fmt.Sprintf("batch=%d", batch), func(b *testing.B) {
			qs := pool[:batch]
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				ix.SearchBatch(qs, k)
			}
			secs := b.Elapsed().Seconds()
			b.ReportMetric(float64(b.N*batch)/secs, "queries/s")
			b.ReportMetric(secs/float64(b.N)*1000, "ms/batch")
		})
	}
}
