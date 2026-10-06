package search

import (
	"fmt"
	"math/rand"
	"testing"

	"vecarena/filter"
)

// randomFilter returns a bitmap holding each of n ids with probability frac.
func randomFilter(r *rand.Rand, n int, frac float64) *filter.Bitmap {
	b := &filter.Bitmap{}
	for i := 0; i < n; i++ {
		if r.Float64() < frac {
			b.Add(uint32(i))
		}
	}
	return b
}

// TestSearchFilteredMatchesBruteForce is the T-07 acceptance test. One batch
// mixes filters on both sides of the planner threshold (sparse and dense
// paths), unfiltered queries, tiny and empty filters, and deleted rows.
func TestSearchFilteredMatchesBruteForce(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	const n, dim, stride, k = 6000, 100, 112, 10
	rows := make([]float32, n*stride)
	for i := 0; i < n; i++ {
		copy(rows[i*stride:], randRows(r, dim))
	}
	deleted := randomFilter(r, n, 0.1)

	fracs := []float64{0.01, 0.04, 0.1, 0.5, 1, -1} // -1: no filter
	var qs [][]float32
	var fs []Filter
	for rep := 0; rep < 3; rep++ {
		for _, f := range fracs {
			qs = append(qs, randRows(r, dim))
			if f < 0 {
				fs = append(fs, nil)
			} else {
				fs = append(fs, randomFilter(r, n, f))
			}
		}
	}
	qs = append(qs, randRows(r, dim), randRows(r, dim), randRows(r, dim))
	fs = append(fs, filter.Of(5, 77, 4000), &filter.Bitmap{}, filter.Of(1, n+10, n+500)) // ids >= n are ignored

	for _, live := range []bool{false, true} {
		ix := New(rows, stride, n)
		ix.TileRows = 333
		if live {
			ix.Live = func(id uint32) bool { return !deleted.Contains(id) }
		}
		got := ix.SearchFiltered(qs, k, fs)
		for i, q := range qs {
			keep := func(id uint32) bool {
				return (fs[i] == nil || fs[i].Contains(id)) && (!live || !deleted.Contains(id))
			}
			want := bruteTopK(rows, stride, n, q, k, keep)
			if len(got[i]) != len(want) {
				t.Fatalf("live=%v q=%d: %d results, want %d", live, i, len(got[i]), len(want))
			}
			for j := range want {
				if got[i][j].ID != want[j] {
					t.Fatalf("live=%v q=%d rank %d: got id %d want %d", live, i, j, got[i][j].ID, want[j])
				}
			}
		}
	}

	if res := New(rows, stride, n).SearchFiltered(qs[:1], k, nil); len(res[0]) != k {
		t.Fatalf("nil filters: %d results", len(res[0]))
	}
	defer func() {
		if recover() == nil {
			t.Fatal("mismatched filters length did not panic")
		}
	}()
	New(rows, stride, n).SearchFiltered(qs, k, fs[:1])
}

func BenchmarkFiltered(b *testing.B) {
	const n, d, k, batch = 100_000, 768, 10, 8
	r := rand.New(rand.NewSource(8))
	ix := New(randRows(r, n*d), d, n)
	qs := make([][]float32, batch)
	for i := range qs {
		qs[i] = randRows(r, d)
	}
	for _, frac := range []float64{0.01, 0.10, 0.50, 1} {
		fs := make([]Filter, batch)
		name := "none"
		if frac < 1 {
			name = fmt.Sprintf("%g%%", frac*100)
			for i := range fs {
				fs[i] = randomFilter(r, n, frac)
			}
		}
		b.Run("selectivity="+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				ix.SearchFiltered(qs, k, fs)
			}
			b.ReportMetric(float64(b.N*batch)/b.Elapsed().Seconds(), "queries/s")
		})
	}
}
