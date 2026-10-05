package search

import (
	"runtime"
	"sort"
	"sync"

	"vecarena/kernel"
)

type Result struct {
	ID    uint32
	Score float32
}

type Index struct {
	rows   []float32
	stride int
	n      int

	TileRows int
	Workers  int

	// Live, if set, reports whether a row may appear in results; rows for
	// which it returns false are skipped. It is called only for scores that
	// would enter a top-k heap. Pass (*vecarena.Arena).IsLive to hide
	// deleted rows, and hold an arena Guard around SearchBatch so rows are
	// not reused mid-scan.
	Live func(id uint32) bool
}

func New(rows []float32, stride, n int) *Index {
	tile := (1 << 20) / (stride * 4)
	if tile < 16 {
		tile = 16
	}
	return &Index{rows: rows, stride: stride, n: n, TileRows: tile, Workers: runtime.GOMAXPROCS(0)}
}

func (ix *Index) SearchBatch(queries [][]float32, k int) [][]Result {
	nq := len(queries)
	if nq == 0 || k <= 0 {
		return make([][]Result, nq)
	}

	groups := (nq + kernel.QueryGroup - 1) / kernel.QueryGroup
	packed := make([]float32, groups*kernel.QueryGroup*ix.stride)
	for i, q := range queries {
		copy(packed[i*ix.stride:], q)
	}

	workers := ix.Workers
	if workers < 1 {
		workers = 1
	}
	tiles := (ix.n + ix.TileRows - 1) / ix.TileRows
	if workers > tiles {
		workers = tiles
	}

	isLive := ix.Live
	partial := make([][]topK, workers)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			heaps := make([]topK, nq)
			for i := range heaps {
				heaps[i] = newTopK(k)
			}
			scores := make([]float32, ix.TileRows*kernel.QueryGroup)

			t0, t1 := w*tiles/workers, (w+1)*tiles/workers
			for t := t0; t < t1; t++ {
				lo := t * ix.TileRows
				hi := min(lo+ix.TileRows, ix.n)
				tile := ix.rows[lo*ix.stride : hi*ix.stride]
				out := scores[:(hi-lo)*kernel.QueryGroup]

				for g := 0; g < groups; g++ {
					qs := packed[g*kernel.QueryGroup*ix.stride : (g+1)*kernel.QueryGroup*ix.stride]
					kernel.DotBatch8(qs, tile, ix.stride, out)

					base := g * kernel.QueryGroup
					live := min(kernel.QueryGroup, nq-base)
					for r := 0; r < hi-lo; r++ {
						row := out[r*kernel.QueryGroup : r*kernel.QueryGroup+live]
						id := uint32(lo + r)
						for q, s := range row {
							h := &heaps[base+q]
							if !h.accepts(s) || (isLive != nil && !isLive(id)) {
								continue
							}
							h.push(id, s)
						}
					}
				}
			}
			partial[w] = heaps
		}(w)
	}
	wg.Wait()

	results := make([][]Result, nq)
	for q := 0; q < nq; q++ {
		final := newTopK(k)
		for w := 0; w < workers; w++ {
			for _, r := range partial[w][q].items {
				final.push(r.ID, r.Score)
			}
		}
		results[q] = final.sorted()
	}
	return results
}

type topK struct {
	k     int
	items []Result
}

func newTopK(k int) topK { return topK{k: k, items: make([]Result, 0, k)} }

// accepts reports whether push(_, s) would change the heap.
func (h *topK) accepts(s float32) bool {
	return len(h.items) < h.k || s > h.items[0].Score
}

func (h *topK) push(id uint32, s float32) {
	if len(h.items) < h.k {
		h.items = append(h.items, Result{id, s})
		h.up(len(h.items) - 1)
		return
	}
	if s <= h.items[0].Score {
		return
	}
	h.items[0] = Result{id, s}
	h.down(0)
}

func (h *topK) up(i int) {
	for i > 0 {
		p := (i - 1) / 2
		if h.items[p].Score <= h.items[i].Score {
			return
		}
		h.items[p], h.items[i] = h.items[i], h.items[p]
		i = p
	}
}

func (h *topK) down(i int) {
	n := len(h.items)
	for {
		l, m := 2*i+1, i
		if l < n && h.items[l].Score < h.items[m].Score {
			m = l
		}
		if r := l + 1; r < n && h.items[r].Score < h.items[m].Score {
			m = r
		}
		if m == i {
			return
		}
		h.items[i], h.items[m] = h.items[m], h.items[i]
		i = m
	}
}

func (h *topK) sorted() []Result {
	out := append([]Result(nil), h.items...)
	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	return out
}
