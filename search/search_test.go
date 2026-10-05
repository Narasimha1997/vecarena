package search

import (
	"context"
	"fmt"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
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

// bruteTopK returns the exact top-k ids in float64. A nil live keeps every row.
func bruteTopK(rows []float32, stride, n int, q []float32, k int, live func(uint32) bool) []uint32 {
	type pair struct {
		id uint32
		s  float64
	}
	all := make([]pair, 0, n)
	for i := 0; i < n; i++ {
		if live != nil && !live(uint32(i)) {
			continue
		}
		var s float64
		for j, x := range q {
			s += float64(x) * float64(rows[i*stride+j])
		}
		all = append(all, pair{uint32(i), s})
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
			want := bruteTopK(rows, stride, n, q, k, nil)
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
			want := bruteTopK(rows, stride, n, q, k, nil)
			for j := range want {
				if got[j].ID != want[j] {
					t.Errorf("rank %d: got %d want %d", j, got[j].ID, want[j])
				}
			}
		}()
	}
	wg.Wait()
}

// TestBatcherCloseRace is the T-03 acceptance test: 1000 concurrent Search
// calls race Close; each must return a result or ErrClosed within 1 s, and a
// second Close must not panic.
func TestBatcherCloseRace(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	const n, stride, k, callers = 2000, 64, 5, 1000
	rows := randRows(r, n*stride)
	b := NewBatcher(New(rows, stride, n), 16, time.Millisecond)

	qs := make([][]float32, callers)
	for i := range qs {
		qs[i] = randRows(r, stride)
	}
	var ok, closed atomic.Int64
	var wg sync.WaitGroup
	start := make(chan struct{})
	for _, q := range qs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			res, err := b.Search(context.Background(), q, k)
			switch {
			case err == ErrClosed:
				closed.Add(1)
			case err != nil:
				t.Errorf("unexpected error: %v", err)
			case len(res) != k:
				t.Errorf("got %d results, want %d", len(res), k)
			default:
				ok.Add(1)
			}
		}()
	}
	close(start)
	time.Sleep(2 * time.Millisecond) // let some batches run before closing
	b.Close()
	b.Close()

	finished := make(chan struct{})
	go func() { wg.Wait(); close(finished) }()
	select {
	case <-finished:
	case <-time.After(time.Second):
		t.Fatalf("Search calls still blocked 1s after Close (%d ok, %d closed)", ok.Load(), closed.Load())
	}
	if _, err := b.Search(context.Background(), qs[0], k); err != ErrClosed {
		t.Fatalf("Search after Close: err = %v, want ErrClosed", err)
	}
	t.Logf("%d results, %d ErrClosed", ok.Load(), closed.Load())
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
