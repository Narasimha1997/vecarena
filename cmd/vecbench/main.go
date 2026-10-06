// Command vecbench is an end-to-end benchmark for vecarena. It fills an arena
// with random vectors, then measures, in order:
//
//   - load:    Add throughput into the off-heap arena
//   - scan:    raw kernel.DotBatch bandwidth over the whole arena
//   - search:  search.SearchBatch throughput and per-batch latency, per batch size
//   - batcher: search.Batcher latency under concurrent single-query clients
//   - recall:  SearchBatch top-k against a brute-force kernel.Dot reference
//
// Each timed phase runs -runs times for -duration; the median run is reported
// with the min–max spread, as SPEC.md section 8 asks for regression numbers.
//
//	go run ./cmd/vecbench -n 100000 -dim 768 -batches 1,8,32,64
package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"text/tabwriter"
	"time"

	"vecarena"
	"vecarena/kernel"
	"vecarena/search"
)

type config struct {
	n, dim, k   int
	batches     []int
	workers     int
	tileRows    int
	duration    time.Duration
	runs        int
	hugePages   bool
	clients     int
	maxBatch    int
	maxWait     time.Duration
	recallQ     int
	seed        uint64
	skipBatcher bool
}

func main() {
	var c config
	var batches string
	flag.IntVar(&c.n, "n", 100_000, "rows in the arena")
	flag.IntVar(&c.dim, "dim", 768, "vector dimension")
	flag.IntVar(&c.k, "k", 10, "top-k per query")
	flag.StringVar(&batches, "batches", "1,8,32,64", "comma-separated SearchBatch sizes")
	flag.IntVar(&c.workers, "workers", 0, "search workers (0 = GOMAXPROCS)")
	flag.IntVar(&c.tileRows, "tile", 0, "rows per cache tile (0 = index default)")
	flag.DurationVar(&c.duration, "duration", 2*time.Second, "length of each timed run")
	flag.IntVar(&c.runs, "runs", 5, "timed runs per configuration (median reported)")
	flag.BoolVar(&c.hugePages, "hugepages", true, "madvise the arena for huge pages")
	flag.IntVar(&c.clients, "clients", 64, "concurrent clients in the batcher phase")
	flag.IntVar(&c.maxBatch, "maxbatch", 64, "batcher max batch size")
	flag.DurationVar(&c.maxWait, "maxwait", time.Millisecond, "batcher max wait before dispatch")
	flag.IntVar(&c.recallQ, "recall", 16, "queries checked against brute force (0 = skip)")
	flag.Uint64Var(&c.seed, "seed", 1, "random seed")
	flag.BoolVar(&c.skipBatcher, "nobatcher", false, "skip the batcher phase")
	flag.Parse()

	var err error
	if c.batches, err = parseInts(batches); err != nil {
		fatal(err)
	}
	if c.n <= 0 || c.dim <= 0 || c.k <= 0 || c.runs <= 0 || c.duration <= 0 {
		fatal(fmt.Errorf("n, dim, k, runs and duration must be > 0"))
	}
	if err := run(c); err != nil {
		fatal(err)
	}
}

func run(c config) error {
	rng := rand.New(rand.NewPCG(c.seed, c.seed^0x9e3779b97f4a7c15))

	fmt.Printf("vecbench  go=%s  %s/%s  cpus=%d  GOMAXPROCS=%d  avx2+fma=%v\n",
		runtime.Version(), runtime.GOOS, runtime.GOARCH, runtime.NumCPU(),
		runtime.GOMAXPROCS(0), kernel.Accelerated())

	// Load.
	a, err := vecarena.New(c.dim, c.n, vecarena.Options{HugePages: c.hugePages})
	if err != nil {
		return err
	}
	defer a.Close()
	v := make([]float32, c.dim)
	start := time.Now()
	for i := 0; i < c.n; i++ {
		fillRandom(rng, v)
		if _, err := a.Add(v); err != nil {
			return err
		}
	}
	load := time.Since(start)
	stride := a.Stride()
	scanBytes := float64(c.n * stride * 4)
	fmt.Printf("arena     n=%d  dim=%d  stride=%d  size=%s  hugepages=%v\n",
		c.n, c.dim, stride, fmtBytes(scanBytes), c.hugePages)
	fmt.Printf("load      %v  (%.0f rows/s)\n\n", load.Round(time.Millisecond), float64(c.n)/load.Seconds())

	rows := a.Block(0, c.n)
	queries := make([][]float32, max(256, slices.Max(c.batches), c.recallQ))
	for i := range queries {
		queries[i] = make([]float32, stride) // zero-padded to stride
		fillRandom(rng, queries[i][:c.dim])
	}

	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', tabwriter.AlignRight)
	defer tw.Flush()

	fmt.Fprintln(tw, "phase\tconfig\tqueries/s (median)\tmin–max\tms/op p50\tp99\tGB/s\t")

	// Scan: one query against every row, single goroutine. The arena stride
	// is always a multiple of 16, as DotBatch requires.
	scores := make([]float32, c.n)
	rs := measure(c, func() int {
		kernel.DotBatch(queries[0], rows, stride, scores)
		return 1
	})
	row(tw, "scan", "DotBatch 1 thread", rs, scanBytes)

	// SearchBatch: whole batches, all workers.
	ix := search.New(rows, stride, c.n)
	if c.workers > 0 {
		ix.Workers = c.workers
	}
	if c.tileRows > 0 {
		ix.TileRows = c.tileRows
	}
	for _, bs := range c.batches {
		qs := queries[:bs]
		rs := measure(c, func() int {
			ix.SearchBatch(qs, c.k)
			return bs
		})
		row(tw, "search", fmt.Sprintf("batch=%d workers=%d", bs, ix.Workers), rs, scanBytes)
	}

	// Batcher: concurrent single-query clients.
	if !c.skipBatcher {
		rs := measureBatcher(c, ix, queries)
		row(tw, "batcher", fmt.Sprintf("clients=%d max=%d wait=%v", c.clients, c.maxBatch, c.maxWait), rs, 0)
	}
	tw.Flush()

	if c.recallQ > 0 {
		fmt.Printf("\nrecall@%d  %.4f over %d queries (vs brute-force kernel.Dot)\n",
			c.k, recall(a, ix, queries[:c.recallQ], c.k), c.recallQ)
	}
	return nil
}

// runStats is one timed run: completed queries, and per-op latencies.
type runStats struct {
	queries int
	elapsed time.Duration
	lat     []time.Duration
}

func (r runStats) qps() float64 { return float64(r.queries) / r.elapsed.Seconds() }

// measure calls op repeatedly for c.duration, c.runs times, after one warm-up
// call. op returns the number of queries it served.
func measure(c config, op func() int) []runStats {
	op()
	out := make([]runStats, c.runs)
	for i := range out {
		runtime.GC()
		var rs runStats
		start := time.Now()
		for time.Since(start) < c.duration {
			t := time.Now()
			rs.queries += op()
			rs.lat = append(rs.lat, time.Since(t))
		}
		rs.elapsed = time.Since(start)
		out[i] = rs
	}
	return out
}

func measureBatcher(c config, ix *search.Index, queries [][]float32) []runStats {
	b := search.NewBatcher(ix, c.maxBatch, c.maxWait)
	// Every client has returned before Close, so B-02 (enqueue racing Close)
	// cannot trigger here.
	defer b.Close()
	ctx := context.Background()

	out := make([]runStats, c.runs)
	for i := range out {
		runtime.GC()
		var stop atomic.Bool
		var wg sync.WaitGroup
		lats := make([][]time.Duration, c.clients)
		start := time.Now()
		for cl := 0; cl < c.clients; cl++ {
			wg.Add(1)
			go func(cl int) {
				defer wg.Done()
				for j := cl; !stop.Load(); j++ {
					t := time.Now()
					if _, err := b.Search(ctx, queries[j%len(queries)], c.k); err != nil {
						return
					}
					lats[cl] = append(lats[cl], time.Since(t))
				}
			}(cl)
		}
		time.Sleep(c.duration)
		stop.Store(true)
		wg.Wait()
		rs := runStats{elapsed: time.Since(start), lat: slices.Concat(lats...)}
		rs.queries = len(rs.lat)
		out[i] = rs
	}
	return out
}

// row prints the median run (by throughput), the min–max spread across runs,
// and latency percentiles pooled over all runs. bytesPerOp > 0 adds the
// effective memory bandwidth, counting one full arena pass per op.
func row(tw *tabwriter.Writer, phase, cfg string, runs []runStats, bytesPerOp float64) {
	slices.SortFunc(runs, func(x, y runStats) int { return compareF(x.qps(), y.qps()) })
	med := runs[len(runs)/2]

	var lat []time.Duration
	ops := 0
	var elapsed time.Duration
	for _, r := range runs {
		lat = append(lat, r.lat...)
		ops += len(r.lat)
		elapsed += r.elapsed
	}
	slices.Sort(lat)

	bw := "-"
	if bytesPerOp > 0 {
		bw = fmt.Sprintf("%.1f", bytesPerOp*float64(ops)/elapsed.Seconds()/1e9)
	}
	fmt.Fprintf(tw, "%s\t%s\t%.1f q/s\t%.1f–%.1f\t%.3f\t%.3f\t%s\t\n",
		phase, cfg, med.qps(), runs[0].qps(), runs[len(runs)-1].qps(),
		ms(pct(lat, 0.50)), ms(pct(lat, 0.99)), bw)
}

// recall returns the mean fraction of the exact top-k ids that SearchBatch
// also returned.
func recall(a *vecarena.Arena, ix *search.Index, qs [][]float32, k int) float64 {
	got := ix.SearchBatch(qs, k)
	dim := a.Dim()
	total := 0.0
	scores := make([]search.Result, 0, a.Len())
	for i, q := range qs {
		scores = scores[:0]
		a.Range(0, a.Len(), func(id uint32, v []float32) {
			scores = append(scores, search.Result{ID: id, Score: kernel.Dot(q[:dim], v)})
		})
		slices.SortFunc(scores, func(x, y search.Result) int { return compareF(y.Score, x.Score) })
		want := make(map[uint32]bool, k)
		for _, r := range scores[:min(k, len(scores))] {
			want[r.ID] = true
		}
		hit := 0
		for _, r := range got[i] {
			if want[r.ID] {
				hit++
			}
		}
		total += float64(hit) / float64(len(want))
	}
	return total / float64(len(qs))
}

func fillRandom(rng *rand.Rand, v []float32) {
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}
}

func pct(sorted []time.Duration, p float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	return sorted[min(len(sorted)-1, int(p*float64(len(sorted))))]
}

func compareF[T float32 | float64](x, y T) int {
	switch {
	case x < y:
		return -1
	case x > y:
		return 1
	}
	return 0
}

func ms(d time.Duration) float64 { return float64(d) / 1e6 }

func fmtBytes(b float64) string {
	switch {
	case b >= 1<<30:
		return fmt.Sprintf("%.2f GiB", b/(1<<30))
	case b >= 1<<20:
		return fmt.Sprintf("%.1f MiB", b/(1<<20))
	}
	return fmt.Sprintf("%.0f KiB", b/(1<<10))
}

func parseInts(s string) ([]int, error) {
	var out []int
	for _, f := range strings.Split(s, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("bad batch size %q", f)
		}
		out = append(out, n)
	}
	return out, nil
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "vecbench:", err)
	os.Exit(1)
}
