package filter

import (
	"fmt"
	"math/rand"
	"slices"
	"testing"
)

type set map[uint32]bool

func (s set) sorted() []uint32 {
	out := make([]uint32, 0, len(s))
	for id := range s {
		out = append(out, id)
	}
	slices.Sort(out)
	return out
}

func check(t *testing.T, name string, b *Bitmap, want set) {
	t.Helper()
	got := b.ToSlice()
	if !slices.Equal(got, want.sorted()) {
		t.Fatalf("%s: got %d ids, want %d", name, len(got), len(want))
	}
	if b.Cardinality() != len(want) {
		t.Fatalf("%s: Cardinality %d, want %d", name, b.Cardinality(), len(want))
	}
	for _, c := range b.conts {
		if c.n == 0 || (c.bits == nil) != (c.n <= arrayMax) {
			t.Fatalf("%s: container n=%d bitset=%v breaks the representation rule", name, c.n, c.bits != nil)
		}
	}
}

// randomSet draws ids from a few chunks with varying density, so containers
// land on both sides of the array/bitset threshold.
func randomSet(r *rand.Rand) set {
	s := set{}
	for chunk := 0; chunk < 4; chunk++ {
		base := uint32(r.Intn(6)) << 16
		count := []int{10, 3000, 5000, 40000}[r.Intn(4)]
		for i := 0; i < count; i++ {
			s[base|uint32(r.Intn(65536))] = true
		}
	}
	return s
}

func fromSet(s set) *Bitmap {
	b := &Bitmap{}
	for id := range s {
		b.Add(id)
	}
	return b
}

func TestBitmapAgainstMap(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for round := 0; round < 12; round++ {
		sa, sb := randomSet(r), randomSet(r)
		a, b := fromSet(sa), fromSet(sb)
		check(t, "a", a, sa)

		and, or, diff := set{}, set{}, set{}
		for id := range sa {
			or[id] = true
			if sb[id] {
				and[id] = true
			} else {
				diff[id] = true
			}
		}
		for id := range sb {
			or[id] = true
		}
		check(t, "Intersect", Intersect(a, b), and)
		check(t, "Union", Union(a, b), or)
		check(t, "Difference", Difference(a, b), diff)

		// Remove most of a clone, forcing bitsets back to arrays and chunks away.
		orig := set{}
		for id := range sa {
			orig[id] = true
		}
		c := a.Clone()
		for id := range sa {
			if r.Intn(10) != 0 {
				c.Remove(id)
				delete(sa, id)
			}
		}
		c.Remove(1 << 31) // absent
		check(t, "after Remove", c, sa)
		check(t, "original after removing from clone", a, orig)
		for id := uint32(0); id < 6<<16; id += 997 {
			if c.Contains(id) != sa[id] {
				t.Fatalf("Contains(%d) = %v", id, !sa[id])
			}
		}
	}
}

func TestAll(t *testing.T) {
	for _, n := range []uint32{0, 1, 63, 64, 4096, 4097, 65535, 65536, 65537, 200000} {
		want := set{}
		for i := uint32(0); i < n; i++ {
			want[i] = true
		}
		check(t, fmt.Sprintf("All(%d)", n), All(n), want)
	}
}

func TestIterateStops(t *testing.T) {
	b := Of(5, 1, 70000, 3)
	var got []uint32
	b.Iterate(func(id uint32) bool { got = append(got, id); return len(got) < 3 })
	if !slices.Equal(got, []uint32{1, 3, 5}) {
		t.Fatalf("got %v", got)
	}
}

// TestColumnsSelect compares every kind of expression against direct
// evaluation over a row-oriented copy of the metadata.
func TestColumnsSelect(t *testing.T) {
	const n = 20000
	type row struct {
		color string
		price float64
		hasC  bool
		hasP  bool
	}
	r := rand.New(rand.NewSource(2))
	c := NewColumns()
	rows := make([]row, n)
	colors := []string{"red", "green", "blue", "black"}
	for i := range rows {
		if r.Intn(10) != 0 {
			rows[i].color, rows[i].hasC = colors[r.Intn(len(colors))], true
			c.SetString(uint32(i), "color", rows[i].color)
		}
		if r.Intn(5) != 0 {
			rows[i].price, rows[i].hasP = float64(r.Intn(1000)), true
			c.SetNumber(uint32(i), "price", rows[i].price)
		}
	}
	// Overwrite and clear some rows to exercise bitmap maintenance.
	for i := 0; i < n; i += 7 {
		rows[i].color, rows[i].hasC = "green", true
		c.SetString(uint32(i), "color", "green")
	}
	for i := 3; i < n; i += 11 {
		rows[i] = row{}
		c.Clear(uint32(i))
	}

	isColor := func(vals ...string) func(row) bool {
		return func(x row) bool { return x.hasC && slices.Contains(vals, x.color) }
	}
	inRange := func(lo, hi float64) func(row) bool {
		return func(x row) bool { return x.hasP && x.price >= lo && x.price <= hi }
	}
	cases := []struct {
		name string
		e    Expr
		want func(row) bool
	}{
		{"Eq", Eq("color", "red"), isColor("red")},
		{"Eq missing value", Eq("color", "purple"), isColor("purple")},
		{"Eq missing column", Eq("size", "L"), func(row) bool { return false }},
		{"In", In("color", "red", "blue"), isColor("red", "blue")},
		{"Range", Range("price", 100, 199), inRange(100, 199)},
		{"AtLeast", AtLeast("price", 990), inRange(990, 1e9)},
		{"AtMost", AtMost("price", 5), inRange(-1e9, 5)},
		{"Has", Has("color"), func(x row) bool { return x.hasC }},
		{"And", And(Eq("color", "green"), Range("price", 0, 499)),
			func(x row) bool { return isColor("green")(x) && inRange(0, 499)(x) }},
		{"Or", Or(Eq("color", "black"), AtLeast("price", 900)),
			func(x row) bool { return isColor("black")(x) || inRange(900, 1e9)(x) }},
		{"Not", Not(Eq("color", "red")), func(x row) bool { return !isColor("red")(x) }},
		{"nested", And(Not(In("color", "red", "blue")), Or(AtMost("price", 50), Not(Has("price")))),
			func(x row) bool {
				return !isColor("red", "blue")(x) && (inRange(-1e9, 50)(x) || !x.hasP)
			}},
		{"empty And", And(), func(row) bool { return true }},
		{"empty Or", Or(), func(row) bool { return false }},
	}
	for _, tc := range cases {
		want := set{}
		for i, x := range rows {
			if tc.want(x) {
				want[uint32(i)] = true
			}
		}
		check(t, tc.name, c.Select(tc.e), want)
	}

	if v, ok := c.String(7, "color"); !ok || v != "green" {
		t.Fatalf("String(7) = %q,%v", v, ok)
	}
	if _, ok := c.String(3, "color"); ok {
		t.Fatal("cleared row still has a color")
	}
	if _, ok := c.Number(3, "price"); ok {
		t.Fatal("cleared row still has a price")
	}
}
