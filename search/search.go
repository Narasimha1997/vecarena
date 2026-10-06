package search

import (
	"runtime"
	"sort"
	"sync"
	"sync/atomic"

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

// Filter restricts one query to a subset of row ids. *filter.Bitmap
// implements it.
type Filter interface {
	Contains(id uint32) bool
	Cardinality() int
	Iterate(fn func(id uint32) bool) // ascending ids, until fn returns false
}

// SparseFraction is the planner threshold. A query whose filter matches fewer
// than this fraction of the index's rows scores only those rows, one by one.
// Any other query joins the tiled batch scan and is filtered at heap push.
const SparseFraction = 0.05

// SearchBatch returns the top k rows for each query, by descending dot
// product. It is SearchFiltered with no filters.
func (ix *Index) SearchBatch(queries [][]float32, k int) [][]Result {
	return ix.SearchFiltered(queries, k, nil)
}

// SearchFiltered is SearchBatch with an optional filter per query: only rows
// in filters[i] can appear in results[i]. filters may be nil, and any entry
// may be nil (no filter); otherwise len(filters) must equal len(queries).
func (ix *Index) SearchFiltered(queries [][]float32, k int, filters []Filter) [][]Result {
	nq := len(queries)
	results := make([][]Result, nq)
	if nq == 0 || k <= 0 {
		return results
	}
	if filters != nil && len(filters) != nq {
		panic("search: len(filters) != len(queries)")
	}

	var dense, sparse []int
	for i, q := range queries {
		if len(q) > ix.stride {
			panic("search: query longer than stride")
		}
		if filters != nil && filters[i] != nil &&
			float64(filters[i].Cardinality()) < SparseFraction*float64(ix.n) {
			sparse = append(sparse, i)
		} else {
			dense = append(dense, i)
		}
	}
	if len(dense) > 0 {
		ix.scanDense(queries, filters, dense, k, results)
	}
	if len(sparse) > 0 {
		ix.scanSparse(queries, filters, sparse, k, results)
	}
	return results
}

// scanDense runs queries[idx] through the tiled DotBatch8 scan over every row
// and stores each result at results[idx[j]].
func (ix *Index) scanDense(queries [][]float32, filters []Filter, idx []int, k int, results [][]Result) {
	nq := len(idx)
	groups := (nq + kernel.QueryGroup - 1) / kernel.QueryGroup
	packed := make([]float32, groups*kernel.QueryGroup*ix.stride)
	// keep[j] combines Live and query j's filter into one predicate (nil if
	// neither applies), so the hot loop has a single call site.
	keep := make([]func(uint32) bool, nq)
	for j, i := range idx {
		copy(packed[j*ix.stride:], queries[i])
		var f Filter
		if filters != nil {
			f = filters[i]
		}
		keep[j] = keepFunc(ix.Live, f)
	}

	workers := ix.Workers
	if workers < 1 {
		workers = 1
	}
	tiles := (ix.n + ix.TileRows - 1) / ix.TileRows
	if workers > tiles {
		workers = tiles
	}

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
					active := min(kernel.QueryGroup, nq-base)
					for r := 0; r < hi-lo; r++ {
						row := out[r*kernel.QueryGroup : r*kernel.QueryGroup+active]
						id := uint32(lo + r)
						for q, s := range row {
							h := &heaps[base+q]
							// Cheapest check first: most scores miss the heap.
							if !h.accepts(s) {
								continue
							}
							if kf := keep[base+q]; kf != nil && !kf(id) {
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

	for j, i := range idx {
		final := newTopK(k)
		for w := 0; w < workers; w++ {
			for _, r := range partial[w][j].items {
				final.push(r.ID, r.Score)
			}
		}
		results[i] = final.sorted()
	}
}

// keepFunc returns a predicate for rows that are live and pass f, or nil if
// every row passes.
func keepFunc(live func(uint32) bool, f Filter) func(uint32) bool {
	switch {
	case f == nil:
		return live
	case live == nil:
		return f.Contains
	}
	return func(id uint32) bool { return f.Contains(id) && live(id) }
}

// scanSparse scores each query in queries[idx] against only the rows its
// filter matches. Queries are spread across workers, one query at a time.
func (ix *Index) scanSparse(queries [][]float32, filters []Filter, idx []int, k int, results [][]Result) {
	isLive := ix.Live
	var next atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < max(1, min(ix.Workers, len(idx))); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				j := int(next.Add(1)) - 1
				if j >= len(idx) {
					return
				}
				q := queries[idx[j]]
				h := newTopK(k)
				filters[idx[j]].Iterate(func(id uint32) bool {
					if int(id) >= ix.n {
						return false // ids ascend, so the rest are out of range too
					}
					off := int(id) * ix.stride
					s := kernel.Dot(q, ix.rows[off:off+len(q)])
					if h.accepts(s) && (isLive == nil || isLive(id)) {
						h.push(id, s)
					}
					return true
				})
				results[idx[j]] = h.sorted()
			}
		}()
	}
	wg.Wait()
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
