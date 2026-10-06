// Package filter provides metadata filtering for vector search: a compressed
// Roaring-style bitmap of row ids, and a column store that evaluates filter
// expressions (Eq, In, Range, And, Or, Not) into bitmaps.
//
// A Bitmap splits the 32-bit id space into chunks of 65536 ids keyed by the
// high 16 bits. Each chunk is a container that is either a sorted array of
// low 16-bit values (when it holds at most 4096 ids, 2 bytes per id) or a
// fixed 8 KiB bitset (when it holds more). Both forms make set operations
// proportional to the data actually present.
package filter

import (
	"math/bits"
	"slices"
)

// arrayMax is the largest array container. Beyond it a bitset (8 KiB) is
// smaller than an array (2 bytes per value).
const arrayMax = 4096

const bitsetWords = 65536 / 64

type container struct {
	arr  []uint16 // sorted values; used when bits is nil
	bits []uint64 // bitsetWords words, or nil
	n    int      // cardinality
}

// Bitmap is a set of uint32 row ids. The zero value is an empty set. A Bitmap
// is not safe for concurrent mutation, but concurrent reads are fine.
type Bitmap struct {
	keys  []uint16 // sorted high 16 bits, one per non-empty container
	conts []*container
}

// Of returns a bitmap holding ids.
func Of(ids ...uint32) *Bitmap {
	b := &Bitmap{}
	for _, id := range ids {
		b.Add(id)
	}
	return b
}

// All returns a bitmap holding every id in [0, n).
func All(n uint32) *Bitmap {
	b := &Bitmap{}
	for hi := uint32(0); hi<<16 < n; hi++ {
		c := &container{bits: make([]uint64, bitsetWords)}
		end := min(n-hi<<16, 65536)
		for w := uint32(0); w < end/64; w++ {
			c.bits[w] = ^uint64(0)
		}
		if r := end % 64; r != 0 {
			c.bits[end/64] = 1<<r - 1
		}
		c.n = int(end)
		if c.n <= arrayMax {
			c.toArray()
		}
		b.keys = append(b.keys, uint16(hi))
		b.conts = append(b.conts, c)
		if hi == 0xffff {
			break
		}
	}
	return b
}

func (b *Bitmap) find(hi uint16) (int, bool) { return slices.BinarySearch(b.keys, hi) }

// Add inserts id.
func (b *Bitmap) Add(id uint32) {
	hi, lo := uint16(id>>16), uint16(id)
	i, ok := b.find(hi)
	if !ok {
		b.keys = slices.Insert(b.keys, i, hi)
		b.conts = slices.Insert(b.conts, i, &container{})
	}
	b.conts[i].add(lo)
}

// Remove deletes id if present.
func (b *Bitmap) Remove(id uint32) {
	i, ok := b.find(uint16(id >> 16))
	if !ok {
		return
	}
	c := b.conts[i]
	c.remove(uint16(id))
	if c.n == 0 {
		b.keys = slices.Delete(b.keys, i, i+1)
		b.conts = slices.Delete(b.conts, i, i+1)
	}
}

// Contains reports whether id is in the set.
func (b *Bitmap) Contains(id uint32) bool {
	i, ok := b.find(uint16(id >> 16))
	return ok && b.conts[i].contains(uint16(id))
}

// Cardinality returns the number of ids in the set.
func (b *Bitmap) Cardinality() int {
	n := 0
	for _, c := range b.conts {
		n += c.n
	}
	return n
}

// Iterate calls fn for each id in ascending order until fn returns false.
func (b *Bitmap) Iterate(fn func(id uint32) bool) {
	for i, c := range b.conts {
		base := uint32(b.keys[i]) << 16
		if c.bits == nil {
			for _, v := range c.arr {
				if !fn(base | uint32(v)) {
					return
				}
			}
			continue
		}
		for w, word := range c.bits {
			for word != 0 {
				t := bits.TrailingZeros64(word)
				if !fn(base | uint32(w*64+t)) {
					return
				}
				word &= word - 1
			}
		}
	}
}

// ToSlice returns the ids in ascending order.
func (b *Bitmap) ToSlice() []uint32 {
	out := make([]uint32, 0, b.Cardinality())
	b.Iterate(func(id uint32) bool { out = append(out, id); return true })
	return out
}

// Clone returns a deep copy.
func (b *Bitmap) Clone() *Bitmap {
	out := &Bitmap{keys: slices.Clone(b.keys), conts: make([]*container, len(b.conts))}
	for i, c := range b.conts {
		out.conts[i] = c.clone()
	}
	return out
}

// Intersect returns the ids in both a and b.
func Intersect(a, b *Bitmap) *Bitmap {
	out := &Bitmap{}
	for i, j := 0, 0; i < len(a.keys) && j < len(b.keys); {
		switch {
		case a.keys[i] < b.keys[j]:
			i++
		case a.keys[i] > b.keys[j]:
			j++
		default:
			out.push(a.keys[i], and(a.conts[i], b.conts[j]))
			i++
			j++
		}
	}
	return out
}

// Union returns the ids in a, b or both.
func Union(a, b *Bitmap) *Bitmap {
	out := &Bitmap{}
	i, j := 0, 0
	for i < len(a.keys) || j < len(b.keys) {
		switch {
		case j == len(b.keys) || (i < len(a.keys) && a.keys[i] < b.keys[j]):
			out.push(a.keys[i], a.conts[i].clone())
			i++
		case i == len(a.keys) || a.keys[i] > b.keys[j]:
			out.push(b.keys[j], b.conts[j].clone())
			j++
		default:
			out.push(a.keys[i], or(a.conts[i], b.conts[j]))
			i++
			j++
		}
	}
	return out
}

// Difference returns the ids in a that are not in b.
func Difference(a, b *Bitmap) *Bitmap {
	out := &Bitmap{}
	j := 0
	for i, k := range a.keys {
		for j < len(b.keys) && b.keys[j] < k {
			j++
		}
		if j < len(b.keys) && b.keys[j] == k {
			out.push(k, andNot(a.conts[i], b.conts[j]))
		} else {
			out.push(k, a.conts[i].clone())
		}
	}
	return out
}

// push appends a container with a key larger than any present, dropping it
// if empty.
func (b *Bitmap) push(k uint16, c *container) {
	if c.n == 0 {
		return
	}
	b.keys = append(b.keys, k)
	b.conts = append(b.conts, c)
}

func (c *container) contains(v uint16) bool {
	if c.bits != nil {
		return c.bits[v/64]&(1<<(v%64)) != 0
	}
	_, ok := slices.BinarySearch(c.arr, v)
	return ok
}

func (c *container) add(v uint16) {
	if c.bits != nil {
		if w, m := &c.bits[v/64], uint64(1)<<(v%64); *w&m == 0 {
			*w |= m
			c.n++
		}
		return
	}
	// Appending in ascending order, the common case when building, is O(1).
	if n := len(c.arr); n == 0 || c.arr[n-1] < v {
		c.arr = append(c.arr, v)
	} else if i, ok := slices.BinarySearch(c.arr, v); !ok {
		c.arr = slices.Insert(c.arr, i, v)
	} else {
		return
	}
	c.n++
	if c.n > arrayMax {
		c.toBits()
	}
}

func (c *container) remove(v uint16) {
	if c.bits != nil {
		if w, m := &c.bits[v/64], uint64(1)<<(v%64); *w&m != 0 {
			*w &^= m
			c.n--
			if c.n <= arrayMax {
				c.toArray()
			}
		}
		return
	}
	if i, ok := slices.BinarySearch(c.arr, v); ok {
		c.arr = slices.Delete(c.arr, i, i+1)
		c.n--
	}
}

func (c *container) toBits() {
	c.bits = make([]uint64, bitsetWords)
	for _, v := range c.arr {
		c.bits[v/64] |= 1 << (v % 64)
	}
	c.arr = nil
}

func (c *container) toArray() {
	arr := make([]uint16, 0, c.n)
	for w, word := range c.bits {
		for word != 0 {
			arr = append(arr, uint16(w*64+bits.TrailingZeros64(word)))
			word &= word - 1
		}
	}
	c.arr, c.bits = arr, nil
}

func (c *container) clone() *container {
	return &container{arr: slices.Clone(c.arr), bits: slices.Clone(c.bits), n: c.n}
}

// fromBits builds a container from bitset words it takes ownership of,
// converting to an array if small enough.
func fromBits(w []uint64) *container {
	n := 0
	for _, x := range w {
		n += bits.OnesCount64(x)
	}
	c := &container{bits: w, n: n}
	if n <= arrayMax {
		c.toArray()
	}
	return c
}

// filterArr keeps the values of arr for which other.contains equals keep.
func filterArr(arr []uint16, other *container, keep bool) *container {
	out := &container{}
	for _, v := range arr {
		if other.contains(v) == keep {
			out.arr = append(out.arr, v)
		}
	}
	out.n = len(out.arr)
	return out
}

func and(a, b *container) *container {
	switch {
	case a.bits == nil:
		return filterArr(a.arr, b, true)
	case b.bits == nil:
		return filterArr(b.arr, a, true)
	}
	w := make([]uint64, bitsetWords)
	for i := range w {
		w[i] = a.bits[i] & b.bits[i]
	}
	return fromBits(w)
}

func or(a, b *container) *container {
	if a.bits == nil && b.bits == nil && len(a.arr)+len(b.arr) <= arrayMax {
		out := &container{arr: make([]uint16, 0, len(a.arr)+len(b.arr))}
		i, j := 0, 0
		for i < len(a.arr) && j < len(b.arr) {
			switch {
			case a.arr[i] < b.arr[j]:
				out.arr = append(out.arr, a.arr[i])
				i++
			case a.arr[i] > b.arr[j]:
				out.arr = append(out.arr, b.arr[j])
				j++
			default:
				out.arr = append(out.arr, a.arr[i])
				i++
				j++
			}
		}
		out.arr = append(append(out.arr, a.arr[i:]...), b.arr[j:]...)
		out.n = len(out.arr)
		return out
	}
	w := make([]uint64, bitsetWords)
	for _, c := range []*container{a, b} {
		if c.bits != nil {
			for i, x := range c.bits {
				w[i] |= x
			}
		} else {
			for _, v := range c.arr {
				w[v/64] |= 1 << (v % 64)
			}
		}
	}
	return fromBits(w)
}

func andNot(a, b *container) *container {
	if a.bits == nil {
		return filterArr(a.arr, b, false)
	}
	w := slices.Clone(a.bits)
	if b.bits != nil {
		for i, x := range b.bits {
			w[i] &^= x
		}
	} else {
		for _, v := range b.arr {
			w[v/64] &^= 1 << (v % 64)
		}
	}
	return fromBits(w)
}
