//go:build !amd64

package kernel

const useAVX2 = false

func dotAVX2(a, b *float32, n int) float32                       { panic("unreachable") }
func l2sqAVX2(a, b *float32, n int) float32                      { panic("unreachable") }
func dotBatchAVX2(q, rows *float32, stride, n int, out *float32) { panic("unreachable") }

func dotBatch8AVX2(qs, rows *float32, stride, n int, out *float32) { panic("unreachable") }
