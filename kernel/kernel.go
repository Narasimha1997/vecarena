package kernel

func Dot(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("kernel: length mismatch")
	}
	if useAVX2 && len(a) >= 16 {
		return dotAVX2(&a[0], &b[0], len(a))
	}
	return dotGeneric(a, b)
}

func L2Sq(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("kernel: length mismatch")
	}
	if useAVX2 && len(a) >= 16 {
		return l2sqAVX2(&a[0], &b[0], len(a))
	}
	return l2sqGeneric(a, b)
}

func DotBatch(q, rows []float32, stride int, out []float32) {
	n := len(out)
	if stride <= 0 || stride%16 != 0 {
		panic("kernel: stride must be a positive multiple of 16")
	}
	if len(q) != stride || len(rows) < n*stride {
		panic("kernel: bad slice lengths")
	}
	if n == 0 {
		return
	}
	if useAVX2 {
		dotBatchAVX2(&q[0], &rows[0], stride, n, &out[0])
		return
	}
	for i := 0; i < n; i++ {
		out[i] = dotGeneric(q, rows[i*stride:(i+1)*stride])
	}
}

func Accelerated() bool { return useAVX2 }

func dotGeneric(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

func l2sqGeneric(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		d0, d1, d2, d3 := a[i]-b[i], a[i+1]-b[i+1], a[i+2]-b[i+2], a[i+3]-b[i+3]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	for ; i < len(a); i++ {
		d := a[i] - b[i]
		s0 += d * d
	}
	return (s0 + s1) + (s2 + s3)
}

const QueryGroup = 8

func DotBatch8(qs, rows []float32, stride int, out []float32) {
	if stride <= 0 || stride%8 != 0 {
		panic("kernel: stride must be a positive multiple of 8")
	}
	n := len(out) / QueryGroup
	if len(out)%QueryGroup != 0 || len(qs) != QueryGroup*stride || len(rows) < n*stride {
		panic("kernel: bad slice lengths")
	}
	if n == 0 {
		return
	}
	if useAVX2 {
		dotBatch8AVX2(&qs[0], &rows[0], stride, n, &out[0])
		return
	}
	for r := 0; r < n; r++ {
		row := rows[r*stride : (r+1)*stride]
		for q := 0; q < QueryGroup; q++ {
			out[r*QueryGroup+q] = dotGeneric(qs[q*stride:(q+1)*stride], row)
		}
	}
}
