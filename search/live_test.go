package search

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"vecarena"
	"vecarena/kernel"
)

// TestDeletedRowsNeverReturned is the T-02 acceptance test: delete the 100
// best matches for a query; none of them may be returned, and the results
// must equal brute force over the remaining rows.
func TestDeletedRowsNeverReturned(t *testing.T) {
	const n, dim, k = 5000, 100, 10
	r := rand.New(rand.NewSource(4))
	a, err := vecarena.New(dim, n, vecarena.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	for i := 0; i < n; i++ {
		a.Add(randRows(r, dim))
	}
	stride := a.Stride()
	rows := a.Block(0, n)
	q := randRows(r, dim)

	deleted := map[uint32]bool{}
	for _, id := range bruteTopK(rows, stride, n, q, 100, nil) {
		a.Delete(id)
		deleted[id] = true
	}

	ix := New(rows, stride, n)
	ix.TileRows = 333
	ix.Live = a.IsLive
	g := a.Enter()
	got := ix.SearchBatch([][]float32{q}, k)[0]
	g.Exit()

	want := bruteTopK(rows, stride, n, q, k, a.IsLive)
	if len(got) != k {
		t.Fatalf("got %d results, want %d", len(got), k)
	}
	for j, res := range got {
		if deleted[res.ID] {
			t.Fatalf("rank %d: deleted id %d returned", j, res.ID)
		}
		if res.ID != want[j] {
			t.Fatalf("rank %d: got id %d want %d", j, res.ID, want[j])
		}
	}
}

// TestCosineTopK is the T-06 acceptance test: with Options.Normalize and a
// normalized query, dot-product search ranks rows by cosine similarity,
// matching a float64 reference over the original, unnormalized vectors.
func TestCosineTopK(t *testing.T) {
	const n, dim, k = 3000, 50, 10
	r := rand.New(rand.NewSource(6))
	a, err := vecarena.New(dim, n, vecarena.Options{Normalize: true})
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	orig := make([][]float32, n)
	for i := range orig {
		v := randRows(r, dim)
		scale := float32(0.01 + 100*r.Float64()) // magnitudes vary by 10^4
		for j := range v {
			v[j] *= scale
		}
		orig[i] = v
		a.Add(v)
	}
	if v := a.Get(0); math.Abs(float64(kernel.Dot(v, v))-1) > 1e-5 {
		t.Fatalf("stored row not unit length: |v|^2 = %v", kernel.Dot(v, v))
	}

	ix := New(a.Block(0, n), a.Stride(), n)
	for qi := 0; qi < 5; qi++ {
		q := randRows(r, dim)
		raw := append([]float32(nil), q...)
		kernel.Normalize(q)
		got := ix.SearchBatch([][]float32{q}, k)[0]

		cos := make([]float64, n)
		ids := make([]int, n)
		for i, v := range orig {
			var dot, nq, nv float64
			for j := range v {
				dot += float64(raw[j]) * float64(v[j])
				nq += float64(raw[j]) * float64(raw[j])
				nv += float64(v[j]) * float64(v[j])
			}
			cos[i] = dot / math.Sqrt(nq*nv)
			ids[i] = i
		}
		sort.Slice(ids, func(x, y int) bool { return cos[ids[x]] > cos[ids[y]] })
		for j := 0; j < k; j++ {
			if int(got[j].ID) != ids[j] {
				t.Fatalf("query %d rank %d: got id %d want %d", qi, j, got[j].ID, ids[j])
			}
			if d := math.Abs(float64(got[j].Score) - cos[ids[j]]); d > 1e-5 {
				t.Fatalf("query %d rank %d: score %v, cosine %v", qi, j, got[j].Score, cos[ids[j]])
			}
		}
	}
}
