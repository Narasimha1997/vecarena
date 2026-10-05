//go:build linux

package search

import (
	"math/rand"
	"testing"

	"vecarena"
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
