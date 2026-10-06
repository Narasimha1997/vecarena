package vecarena

import "sync"

// Collection pairs an Arena with external keys of any comparable type, such
// as string or uint64. Each live key maps to exactly one row id, and search
// results (row ids) map back to keys with Key.
//
// All writes must go through the Collection. Calling Add or Delete on the
// underlying Arena directly breaks the mapping.
type Collection[K comparable] struct {
	a *Arena

	mu   sync.RWMutex // guards ids and keys; held for writing around arena writes
	ids  map[K]uint32
	keys []K // indexed by row id; the zero K for deleted rows
}

// NewCollection creates a Collection backed by a new Arena.
func NewCollection[K comparable](dim, maxRows int, opt Options) (*Collection[K], error) {
	a, err := New(dim, maxRows, opt)
	if err != nil {
		return nil, err
	}
	return &Collection[K]{a: a, ids: make(map[K]uint32)}, nil
}

// Arena returns the underlying arena, for searching (Block, Stride, IsLive,
// Enter). Do not write to it directly.
func (c *Collection[K]) Arena() *Arena { return c.a }

// Upsert stores v under key and returns its row id. If key already exists,
// the new vector goes into a new row and the old row is deleted, rather than
// being overwritten in place, so concurrent readers never see a torn row.
// The returned id may therefore differ from the key's previous id. On error
// (dimension mismatch, arena full) the previous value is kept.
func (c *Collection[K]) Upsert(key K, v []float32) (uint32, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	id, err := c.a.Add(v)
	if err != nil {
		return 0, err
	}
	old, existed := c.ids[key]
	c.ids[key] = id
	if int(id) >= len(c.keys) {
		c.keys = append(c.keys, make([]K, int(id)+1-len(c.keys))...)
	}
	c.keys[id] = key
	// The new row is mapped before the old one is deleted, so the key is
	// never missing for a reader.
	if existed {
		c.drop(old)
	}
	return id, nil
}

// Delete removes key and its vector. It reports whether key was present.
func (c *Collection[K]) Delete(key K) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	id, ok := c.ids[key]
	if !ok {
		return false
	}
	delete(c.ids, key)
	c.drop(id)
	return true
}

// drop deletes row id and clears its key. Caller holds mu for writing.
func (c *Collection[K]) drop(id uint32) {
	var zero K
	c.keys[id] = zero // release the key (for example a string) to the GC
	c.a.Delete(id)
}

// ID returns the row id currently holding key.
func (c *Collection[K]) ID(key K) (uint32, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	id, ok := c.ids[key]
	return id, ok
}

// Key returns the key stored in row id, or false if the row is not live.
// To translate search results reliably, call it while holding the arena
// Guard used for the search, so no returned id can be reused in between.
func (c *Collection[K]) Key(id uint32) (K, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	if int(id) >= len(c.keys) || !c.a.IsLive(id) {
		var zero K
		return zero, false
	}
	return c.keys[id], true
}

// Get returns a view of the vector stored under key. The view is valid until
// the key is upserted or deleted; hold an arena Guard if writers run
// concurrently.
func (c *Collection[K]) Get(key K) ([]float32, bool) {
	id, ok := c.ID(key)
	if !ok {
		return nil, false
	}
	return c.a.Get(id), true
}

// Len returns the number of keys.
func (c *Collection[K]) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.ids)
}

// Close releases the arena. It must not run concurrently with any other call.
func (c *Collection[K]) Close() error { return c.a.Close() }
