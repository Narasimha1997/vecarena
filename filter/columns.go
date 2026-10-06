package filter

import (
	"math"
	"sync"
)

// Columns stores per-row metadata in named columns and evaluates filter
// expressions over it. Row ids are the same ids the vectors use.
//
// String columns are dictionary encoded: each distinct value gets a code,
// each row stores a code, and each code keeps a Bitmap of its rows, so Eq and
// In are bitmap lookups. Number columns store a float64 per row and Range
// scans the column.
//
// Columns is safe for concurrent use. Select returns a new Bitmap that later
// writes do not affect.
type Columns struct {
	mu   sync.RWMutex
	n    uint32 // 1 + the highest row id ever set; the universe for Not
	strs map[string]*strColumn
	nums map[string]*numColumn
}

type strColumn struct {
	dict  map[string]uint32 // value -> code; codes start at 1
	vals  []string          // code -> value; vals[0] is unused
	codes []uint32          // row -> code; 0 means unset
	rows  []*Bitmap         // code -> rows holding it; rows[0] is unused
}

type numColumn struct {
	vals []float64
	set  Bitmap // rows that have a value
}

// NewColumns returns an empty column store.
func NewColumns() *Columns {
	return &Columns{strs: map[string]*strColumn{}, nums: map[string]*numColumn{}}
}

func (c *Columns) grow(id uint32) {
	c.n = max(c.n, id+1)
}

// SetString sets column col of row id to val, replacing any previous value.
func (c *Columns) SetString(id uint32, col, val string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.grow(id)
	sc := c.strs[col]
	if sc == nil {
		sc = &strColumn{dict: map[string]uint32{}, vals: []string{""}, rows: []*Bitmap{nil}}
		c.strs[col] = sc
	}
	code, ok := sc.dict[val]
	if !ok {
		code = uint32(len(sc.rows))
		sc.dict[val] = code
		sc.vals = append(sc.vals, val)
		sc.rows = append(sc.rows, &Bitmap{})
	}
	if int(id) >= len(sc.codes) {
		sc.codes = append(sc.codes, make([]uint32, int(id)+1-len(sc.codes))...)
	}
	if old := sc.codes[id]; old != 0 {
		sc.rows[old].Remove(id)
	}
	sc.codes[id] = code
	sc.rows[code].Add(id)
}

// SetNumber sets column col of row id to v, replacing any previous value.
func (c *Columns) SetNumber(id uint32, col string, v float64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.grow(id)
	nc := c.nums[col]
	if nc == nil {
		nc = &numColumn{}
		c.nums[col] = nc
	}
	if int(id) >= len(nc.vals) {
		nc.vals = append(nc.vals, make([]float64, int(id)+1-len(nc.vals))...)
	}
	nc.vals[id] = v
	nc.set.Add(id)
}

// String returns the value of string column col for row id.
func (c *Columns) String(id uint32, col string) (string, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	sc := c.strs[col]
	if sc == nil || int(id) >= len(sc.codes) || sc.codes[id] == 0 {
		return "", false
	}
	return sc.vals[sc.codes[id]], true
}

// Number returns the value of number column col for row id.
func (c *Columns) Number(id uint32, col string) (float64, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	nc := c.nums[col]
	if nc == nil || !nc.set.Contains(id) {
		return 0, false
	}
	return nc.vals[id], true
}

// Clear removes every value of row id, for example when the row is deleted
// and its id may be reused.
func (c *Columns) Clear(id uint32) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, sc := range c.strs {
		if int(id) < len(sc.codes) && sc.codes[id] != 0 {
			sc.rows[sc.codes[id]].Remove(id)
			sc.codes[id] = 0
		}
	}
	for _, nc := range c.nums {
		nc.set.Remove(id)
	}
}

// Select evaluates e and returns the matching row ids.
func (c *Columns) Select(e Expr) *Bitmap {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return e.eval(c)
}

// Expr is a filter expression, built with Eq, In, Range, Has, And, Or and Not.
type Expr interface {
	eval(c *Columns) *Bitmap
}

type eqExpr struct {
	col  string
	vals []string
}

type rangeExpr struct {
	col    string
	lo, hi float64
}

type hasExpr struct{ col string }

type boolExpr struct {
	op   byte // '&' or '|'
	args []Expr
}

type notExpr struct{ e Expr }

// Eq matches rows whose string column col equals val.
func Eq(col, val string) Expr { return eqExpr{col, []string{val}} }

// In matches rows whose string column col equals any of vals.
func In(col string, vals ...string) Expr { return eqExpr{col, vals} }

// Range matches rows whose number column col lies in [lo, hi]. Use
// math.Inf for an open end.
func Range(col string, lo, hi float64) Expr { return rangeExpr{col, lo, hi} }

// AtLeast matches rows whose number column col is >= lo.
func AtLeast(col string, lo float64) Expr { return rangeExpr{col, lo, math.Inf(1)} }

// AtMost matches rows whose number column col is <= hi.
func AtMost(col string, hi float64) Expr { return rangeExpr{col, math.Inf(-1), hi} }

// Has matches rows that have any value in column col.
func Has(col string) Expr { return hasExpr{col} }

// And matches rows that match every argument. And() matches every row.
func And(es ...Expr) Expr { return boolExpr{'&', es} }

// Or matches rows that match any argument. Or() matches no row.
func Or(es ...Expr) Expr { return boolExpr{'|', es} }

// Not matches every row id below the highest id ever set that e does not
// match, including rows with no value in e's columns.
func Not(e Expr) Expr { return notExpr{e} }

func (e eqExpr) eval(c *Columns) *Bitmap {
	out := &Bitmap{}
	sc := c.strs[e.col]
	if sc == nil {
		return out
	}
	for _, v := range e.vals {
		if code, ok := sc.dict[v]; ok {
			out = Union(out, sc.rows[code])
		}
	}
	return out
}

func (e rangeExpr) eval(c *Columns) *Bitmap {
	out := &Bitmap{}
	nc := c.nums[e.col]
	if nc == nil {
		return out
	}
	nc.set.Iterate(func(id uint32) bool {
		if v := nc.vals[id]; v >= e.lo && v <= e.hi {
			out.Add(id) // ascending ids: appends, O(1) each
		}
		return true
	})
	return out
}

func (e hasExpr) eval(c *Columns) *Bitmap {
	out := &Bitmap{}
	if sc := c.strs[e.col]; sc != nil {
		for _, b := range sc.rows[1:] {
			out = Union(out, b)
		}
	}
	if nc := c.nums[e.col]; nc != nil {
		out = Union(out, &nc.set)
	}
	return out
}

func (e boolExpr) eval(c *Columns) *Bitmap {
	if len(e.args) == 0 {
		if e.op == '&' {
			return All(c.n)
		}
		return &Bitmap{}
	}
	out := e.args[0].eval(c)
	for _, a := range e.args[1:] {
		if e.op == '&' {
			if out.Cardinality() == 0 {
				return out
			}
			out = Intersect(out, a.eval(c))
		} else {
			out = Union(out, a.eval(c))
		}
	}
	return out
}

func (e notExpr) eval(c *Columns) *Bitmap {
	return Difference(All(c.n), e.e.eval(c))
}
