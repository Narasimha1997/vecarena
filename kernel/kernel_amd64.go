//go:build amd64

package kernel

//go:noescape
func dotAVX2(a, b *float32, n int) float32

//go:noescape
func l2sqAVX2(a, b *float32, n int) float32

//go:noescape
func dotBatchAVX2(q, rows *float32, stride, n int, out *float32)

func cpuid(leaf, subleaf uint32) (eax, ebx, ecx, edx uint32)
func xgetbv() (eax, edx uint32)

var useAVX2 = detectAVX2FMA()

func detectAVX2FMA() bool {
	maxLeaf, _, _, _ := cpuid(0, 0)
	if maxLeaf < 7 {
		return false
	}
	_, _, ecx1, _ := cpuid(1, 0)
	fma := ecx1&(1<<12) != 0
	osxsave := ecx1&(1<<27) != 0
	avx := ecx1&(1<<28) != 0
	if !fma || !osxsave || !avx {
		return false
	}
	xcr0, _ := xgetbv()
	if xcr0&0b110 != 0b110 {
		return false
	}
	_, ebx7, _, _ := cpuid(7, 0)
	return ebx7&(1<<5) != 0
}

//go:noescape
func dotBatch8AVX2(qs, rows *float32, stride, n int, out *float32)
