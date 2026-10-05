package kernel

import (
	"math"
	"math/rand"
	"testing"
)

func randVec(r *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = r.Float32()*2 - 1
	}
	return v
}

func dot64(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

func l2sq64(a, b []float32) float64 {
	var s float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		s += d * d
	}
	return s
}

func close(got float32, want float64, n int) bool {
	tol := 1e-5*float64(n) + 1e-4*math.Abs(want)
	return math.Abs(float64(got)-want) <= tol
}

func TestDetect(t *testing.T) { t.Logf("assembly kernels in use: %v", Accelerated()) }

func TestDotAndL2AllLengths(t *testing.T) {
	r := rand.New(rand.NewSource(1))

	for n := 0; n <= 200; n++ {
		a, b := randVec(r, n), randVec(r, n)
		if n == 0 {
			if Dot(a, b) != 0 || L2Sq(a, b) != 0 {
				t.Fatal("empty input")
			}
			continue
		}
		if got, want := Dot(a, b), dot64(a, b); !close(got, want, n) {
			t.Fatalf("Dot n=%d: got %v want %v", n, got, want)
		}
		if got, want := L2Sq(a, b), l2sq64(a, b); !close(got, want, n) {
			t.Fatalf("L2Sq n=%d: got %v want %v", n, got, want)
		}
	}
	for _, n := range []int{384, 768, 1024, 1536, 3072} {
		a, b := randVec(r, n), randVec(r, n)
		if got, want := Dot(a, b), dot64(a, b); !close(got, want, n) {
			t.Fatalf("Dot n=%d: got %v want %v", n, got, want)
		}
	}
}

func TestDotBatch(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	for _, stride := range []int{16, 32, 48, 784} {
		const rows = 37
		q := randVec(r, stride)
		block := randVec(r, rows*stride)
		out := make([]float32, rows)
		DotBatch(q, block, stride, out)
		for i := 0; i < rows; i++ {
			want := dot64(q, block[i*stride:(i+1)*stride])
			if !close(out[i], want, stride) {
				t.Fatalf("stride=%d row=%d: got %v want %v", stride, i, out[i], want)
			}
		}
	}
}

var sink float32

func benchPair(b *testing.B, f func(a, b []float32) float32, n int) {
	r := rand.New(rand.NewSource(3))
	x, y := randVec(r, n), randVec(r, n)
	b.SetBytes(int64(2 * 4 * n))
	for i := 0; i < b.N; i++ {
		sink = f(x, y)
	}
}

func BenchmarkDot768Generic(b *testing.B)  { benchPair(b, dotGeneric, 768) }
func BenchmarkDot768(b *testing.B)         { benchPair(b, Dot, 768) }
func BenchmarkL2Sq768Generic(b *testing.B) { benchPair(b, l2sqGeneric, 768) }
func BenchmarkL2Sq768(b *testing.B)        { benchPair(b, L2Sq, 768) }

func BenchmarkDotBatch100k768(b *testing.B) {
	const n, d = 100_000, 768
	r := rand.New(rand.NewSource(4))
	block := randVec(r, n*d)
	q := randVec(r, d)
	out := make([]float32, n)
	b.SetBytes(int64(n * d * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		DotBatch(q, block, d, out)
	}
}

func BenchmarkLoopDot100k768(b *testing.B) {
	const n, d = 100_000, 768
	r := rand.New(rand.NewSource(4))
	block := randVec(r, n*d)
	q := randVec(r, d)
	out := make([]float32, n)
	b.SetBytes(int64(n * d * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := 0; j < n; j++ {
			out[j] = dotGeneric(q, block[j*d:(j+1)*d])
		}
	}
}

func TestDotBatch8(t *testing.T) {
	r := rand.New(rand.NewSource(5))
	for _, stride := range []int{8, 16, 64, 784} {
		const rows = 29
		qs := randVec(r, QueryGroup*stride)
		block := randVec(r, rows*stride)
		out := make([]float32, rows*QueryGroup)
		DotBatch8(qs, block, stride, out)
		for i := 0; i < rows; i++ {
			for q := 0; q < QueryGroup; q++ {
				want := dot64(qs[q*stride:(q+1)*stride], block[i*stride:(i+1)*stride])
				if !close(out[i*QueryGroup+q], want, stride) {
					t.Fatalf("stride=%d row=%d q=%d: got %v want %v", stride, i, q, out[i*QueryGroup+q], want)
				}
			}
		}
	}
}
