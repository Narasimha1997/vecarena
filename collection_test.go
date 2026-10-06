package vecarena

import (
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

// TestCollectionRoundTrip runs random upserts and deletes against a reference
// map and checks that keys, ids and vectors always round-trip.
func TestCollectionRoundTrip(t *testing.T) {
	const dim, keys, ops = 8, 300, 20000
	c, err := NewCollection[string](dim, keys*2, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	ref := map[string]float32{} // key -> value written to v[0]
	r := rand.New(rand.NewSource(1))
	v := make([]float32, dim)
	for i := 0; i < ops; i++ {
		key := fmt.Sprintf("doc-%d", r.Intn(keys))
		if r.Intn(4) == 0 {
			_, had := ref[key]
			if c.Delete(key) != had {
				t.Fatalf("op %d: Delete(%q) reported %v, want %v", i, key, !had, had)
			}
			delete(ref, key)
			continue
		}
		v[0] = float32(i)
		id, err := c.Upsert(key, v)
		if err != nil {
			t.Fatalf("op %d: %v", i, err)
		}
		ref[key] = float32(i)
		if k, ok := c.Key(id); !ok || k != key {
			t.Fatalf("op %d: Key(%d) = %q,%v want %q", i, id, k, ok, key)
		}
	}

	if c.Len() != len(ref) || c.Arena().Len() != len(ref) {
		t.Fatalf("Len = %d, arena Len = %d, want %d", c.Len(), c.Arena().Len(), len(ref))
	}
	for key, want := range ref {
		id, ok := c.ID(key)
		if !ok {
			t.Fatalf("key %q missing", key)
		}
		if k, ok := c.Key(id); !ok || k != key {
			t.Fatalf("Key(ID(%q)) = %q,%v", key, k, ok)
		}
		if got, _ := c.Get(key); got[0] != want {
			t.Fatalf("Get(%q)[0] = %v want %v", key, got[0], want)
		}
	}
	live := 0
	c.Arena().Range(0, keys*2, func(id uint32, _ []float32) {
		if _, ok := c.Key(id); !ok {
			t.Fatalf("live row %d has no key", id)
		}
		live++
	})
	if live != len(ref) {
		t.Fatalf("%d live rows, want %d (old rows leaked by Upsert?)", live, len(ref))
	}
	if _, ok := c.ID("missing"); ok {
		t.Fatal("ID of a missing key succeeded")
	}
	if c.Delete("missing") {
		t.Fatal("Delete of a missing key reported true")
	}
	if _, err := c.Upsert("bad", make([]float32, dim+1)); err == nil {
		t.Fatal("wrong dimension accepted")
	}
}

// TestCollectionConcurrentReaders translates row ids to keys while a writer
// keeps upserting. Each row's vector encodes its key, so a mismatch means the
// mapping and the arena disagreed. Must pass -race.
func TestCollectionConcurrentReaders(t *testing.T) {
	const dim, keys = 16, 200
	c, _ := NewCollection[uint64](dim, keys*4, Options{})
	defer c.Close()
	v := make([]float32, dim)
	for k := uint64(0); k < keys; k++ {
		v[0] = float32(k)
		c.Upsert(k, v)
	}

	var stop atomic.Bool
	var wg sync.WaitGroup
	for r := 0; r < 4; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			a := c.Arena()
			for !stop.Load() {
				g := a.Enter()
				a.Range(0, keys*4, func(id uint32, row []float32) {
					if k, ok := c.Key(id); ok && float32(k) != row[0] {
						t.Errorf("row %d holds %v but maps to key %d", id, row[0], k)
					}
				})
				g.Exit()
			}
		}()
	}
	r := rand.New(rand.NewSource(2))
	for i := 0; i < 20000; i++ {
		k := uint64(r.Intn(keys))
		v[0] = float32(k)
		if _, err := c.Upsert(k, v); err != nil {
			runtime.Gosched() // arena full until readers release deleted rows
		}
	}
	stop.Store(true)
	wg.Wait()
}

// TestCollectionMemoryPerKey measures the Go heap used by the key mapping
// (not the vectors, which live in the arena). The numbers are recorded in
// SPEC.md section 4.6.
func TestCollectionMemoryPerKey(t *testing.T) {
	if testing.Short() {
		t.Skip("allocates 1M keys")
	}
	const n = 1_000_000
	measure := func(name string, upsert func(c any, i int) any) {
		var before, after runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&before)
		c := upsert(nil, -1)
		for i := 0; i < n; i++ {
			upsert(c, i)
		}
		runtime.GC()
		runtime.ReadMemStats(&after)
		t.Logf("%s: %.1f bytes of Go heap per key", name, float64(after.HeapAlloc-before.HeapAlloc)/n)
		runtime.KeepAlive(c)
	}
	v := make([]float32, 1)
	measure("uint64", func(c any, i int) any {
		if i < 0 {
			c, _ := NewCollection[uint64](1, n, Options{})
			return c
		}
		c.(*Collection[uint64]).Upsert(uint64(i), v)
		return c
	})
	measure("string (16 bytes)", func(c any, i int) any {
		if i < 0 {
			c, _ := NewCollection[string](1, n, Options{})
			return c
		}
		c.(*Collection[string]).Upsert(fmt.Sprintf("key-%012d", i), v)
		return c
	})
}
