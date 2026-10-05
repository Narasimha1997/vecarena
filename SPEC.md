# vecarena: implementation spec

An in-memory, brute-force (flat) vector search engine in Go, with no dependencies outside the
standard library. This document is the source of truth: every public API, invariant, algorithm,
test and known bug. Code that disagrees with this spec is a bug in one or the other; fix whichever
is wrong and keep them in sync.

Status markers used below:

- **[DONE]**: implemented in the reference code and covered by tests.
- **[BUG]**: implemented, but wrong in a known way. Fix before building on it.
- **[TODO]**: not implemented yet. The backlog (section 9) gives the order and acceptance criteria.

---

## 1. Goals and non-goals

**Goals**
- Exact (brute-force) top-k search over float32 vectors held entirely in RAM.
- Similarity is the **dot product**; cosine similarity is supported by L2-normalising vectors
  before insert (a helper is on the backlog, `T-06`).
- The limiting factor should be memory bandwidth: SIMD kernels, batching, cache tiling, all cores.
- Predictable memory: vectors live off the Go heap, so the garbage collector (GC) never scans them.
- Safe concurrency: one writer at a time, many readers, correct under `go test -race`.

**Non-goals (for now)**
- Approximate indexes (HNSW, IVF). The project deliberately chose a flat buffer.
- Durability, distribution, multi-tenancy. Durability is on the backlog (`T-10`).

## 2. Environment

| Item | Requirement |
|---|---|
| Go | 1.24 or newer (uses generics, `min`/`max`/`clear` builtins, `atomic.Uint64`) |
| Primary target | linux/amd64 with AVX2 and FMA (Intel Haswell+, AMD Zen+) |
| Must compile | linux/arm64, using the portable Go fallbacks |
| Module path | `vecarena` |
| Dependencies | standard library only. `golang.org/x/sys` is allowed if network access permits; the reference code avoids it. |

## 3. Repository layout

```
vecarena/
├── go.mod
├── SPEC.md                 # this file
├── CLAUDE.md               # working rules for Claude Code
├── arena_linux.go          # package vecarena: off-heap vector arena (linux only)
├── arena_test.go
├── kernel/                 # distance kernels
│   ├── kernel.go           # public API + portable fallbacks
│   ├── kernel_amd64.go     # asm declarations + CPU feature detection
│   ├── kernel_amd64.s      # AVX2+FMA assembly
│   ├── kernel_other.go     # !amd64 stubs
│   └── kernel_test.go
├── cmd/vecbench/           # end-to-end benchmark CLI (linux only)
│   └── main.go
├── search/                 # batched top-k search + request batcher
│   ├── search.go
│   ├── batcher.go
│   ├── search_test.go
│   └── live_linux_test.go  # search over a real arena with deleted rows
└── spsc/                   # single-producer single-consumer ring buffers
    ├── spsc.go
    ├── spsc_test.go
    └── spin_test.go        # busy-spin and batched-channel benchmarks
```

## 4. Package `vecarena` (arena)

Stores fixed-dimension float32 vectors in one off-heap memory region with a fixed row stride.

### 4.1 Memory layout
- `stride = ceil(dim / 16) * 16` floats, so every row is a multiple of 64 bytes (one cache line).
- One anonymous `mmap` of `stride * maxRows * 4` bytes with
  `PROT_READ|PROT_WRITE`, `MAP_PRIVATE|MAP_ANON|MAP_NORESERVE`. RAM is committed lazily,
  page by page, on first write. The base address never moves.
- Row `id` occupies floats `[id*stride, id*stride + stride)`. The base is page-aligned, so every
  row is 64-byte aligned.
- **Invariant P (padding):** floats `[dim, stride)` of every row are always zero. `mmap` zero-fills
  memory and `Add` writes only `dim` floats. Kernels depend on this to scan whole strides.
- No Go pointers are ever stored in arena memory.

### 4.2 Types

```go
type Options struct {
    HugePages bool // madvise(MADV_HUGEPAGE), best effort; errors ignored
    Lock      bool // mlock the whole region; fails if RLIMIT_MEMLOCK is too low
}
type Arena struct { /* unexported */ }
```

Internal state: `count` (atomic int64, high-water mark of allocated ids), `live` (atomic int64),
`wmu` (mutex serialising writers), `dead` (tombstone bitset, one bit per row, accessed atomically),
`free` (LIFO stack of reusable ids, guarded by `wmu`), `pending` (FIFO of deleted ids tagged with
their delete epoch, guarded by `wmu`), `epoch` (atomic uint64, written only under `wmu`) and
`readers[2]` (cache-line-padded atomic reader counts, indexed by `epoch & 1`).

### 4.3 API [DONE]

| Signature | Semantics |
|---|---|
| `New(dim, maxRows int, opt Options) (*Arena, error)` | Errors: `dim <= 0` or `maxRows <= 0`; mmap failure; mlock failure (the region is unmapped before returning). |
| `(*Arena) Add(v []float32) (uint32, error)` | `len(v) != dim` returns error `"vecarena: wrong dimension"`. Reuses the most recently freed id (LIFO) if one exists: copies `v`, then atomically clears its dead bit. Otherwise takes id `count`, copies `v`, then atomically stores `count+1` (publishing the row). If `count == maxRows` with no free ids, returns error `"vecarena: full"`. Increments `live`. |
| `(*Arena) Delete(id uint32)` | No-op if `id >= count` or the row is already dead. Otherwise sets the dead bit atomically, appends `(id, epoch)` to `pending`, decrements `live`, then runs reclamation (4.4). Idempotent. The id becomes reusable only once reclaimed. |
| `(*Arena) Enter() Guard`, `(Guard) Exit()` | Mark a read section. While a `Guard` is held, no row that was live when it was taken is reused. Every `Enter` must be paired with `Exit`. |
| `(*Arena) IsLive(id uint32) bool` | `id < count` and the dead bit is clear. Lock-free; usable as `search.Index.Live`. |
| `(*Arena) Get(id uint32) []float32` | View of length `dim` (capacity `stride`). No liveness check: a dead id returns stale data. Out-of-range ids panic (slice bounds). Hold a `Guard` while using the view if writers run concurrently. |
| `(*Arena) Dim() int`, `Stride() int`, `Len() int` | `Len` is the live row count. |
| `(*Arena) Block(lo, hi int) []float32` | Contiguous view of rows `[lo, hi)`, including dead rows. Rows `>= count` are zero. This is the input format for `kernel` and `search`. Hold a `Guard` while scanning it if writers run concurrently. |
| `(*Arena) Range(lo, hi int, fn func(id uint32, v []float32))` | Calls `fn` for each live row in `[lo, min(hi, count))`, in ascending id order. Skips 64 dead rows at a time using whole bitset words. Holds a `Guard` for the whole scan. |
| `(*Arena) Close() error` | Unmaps the region. Must not run concurrently with any other call. Every view obtained earlier becomes invalid. |

### 4.4 Concurrency contract
- Writers (`Add`, `Delete`) are serialised by `wmu`, so any number of goroutines may call them.
- Readers (`Get`, `Block`, `Range`) take no lock and may run concurrently with writers.
- A reader sees a row only after it has been fully written, because `count` is published after the copy.
- **Epoch-based reclamation [DONE] (fixed B-01, torn read on slot reuse).** A reader that checked
  the dead bit before a `Delete` may still be reading that row, so a deleted id is not reused until
  every reader that could see it has exited:
  - `Enter`: load `e = epoch`, increment `readers[e&1]`, re-load `epoch`; if it changed, decrement
    and retry. Only epochs `E` and `E-1` can have readers.
  - Reclamation (under `wmu`, from `Delete`, and from `Add` when `free` is empty): move every
    pending id with `deleteEpoch + 2 <= epoch` to `free`, in delete order; otherwise, if
    `readers[(epoch+1)&1]` (the readers of `epoch-1`) is zero, increment `epoch` and repeat. It never
    blocks; with no readers, a deleted id is reusable immediately.
  - Readers that enter after a delete see the dead bit, so they never read the old row. Readers of
    the delete epoch or earlier are gone once `epoch` has advanced twice.
  - A long-held `Guard` delays reuse; `Add` returns `"vecarena: full"` if no id can be reclaimed.

### 4.5 Tests [DONE]
- `TestAddGetDeleteReuse`: stride for dim 3 is 16; rows are 64-byte aligned; `Range` skips dead
  rows; a freed id is reused; `Len` is correct.
- `TestRangeSkipsWholeWords`: 300 rows, delete 0–199; `Range` yields exactly 200–299 with correct data.
- `TestConcurrentReadersOneWriter`: 4 readers running `Range` while one writer adds 50k rows and
  deletes some; no row may be torn. Must pass `-race`.
- `TestNoTornReadsOnSlotReuse`: 4 readers check `v[j] == v[0]+j` on every row while a writer
  deletes random rows and re-adds new ones for 10 s (1 s with `-short`); zero torn rows, and slots
  must actually be reused. Fails within 1 s if reclamation ignores epochs.
- `BenchmarkScan100k768`: scalar scan via `Range` (baseline only).

## 5. Package `kernel` (distance kernels)

### 5.1 API [DONE]

| Signature | Preconditions (violations panic) | Result |
|---|---|---|
| `Dot(a, b []float32) float32` | `len(a) == len(b)` | `a·b`; 0 for empty input |
| `L2Sq(a, b []float32) float32` | `len(a) == len(b)` | squared Euclidean distance (no sqrt; ranking is unchanged) |
| `DotBatch(q, rows []float32, stride int, out []float32)` | `stride > 0`, `stride % 16 == 0`, `len(q) == stride`, `len(rows) >= len(out)*stride` | `out[i] = q · rows[i*stride:(i+1)*stride]`, with `n = len(out)` |
| `DotBatch8(qs, rows []float32, stride int, out []float32)` | `stride > 0`, `stride % 8 == 0`, `len(out) % 8 == 0`, `len(qs) == 8*stride`, `len(rows) >= (len(out)/8)*stride` | `out[r*8+q] = qs[q*stride:(q+1)*stride] · row_r` (row-major) |
| `const QueryGroup = 8` | | queries per `DotBatch8` pass |
| `Accelerated() bool` | | whether the AVX2 assembly paths are active |

Callers zero-pad queries and rows out to `stride`; Invariant P guarantees this for arena rows.

### 5.2 Dispatch
- `useAVX2` is computed once at package init: CPUID leaf 1 ECX bits 12 (FMA), 27 (OSXSAVE)
  and 28 (AVX); XGETBV(0) bits 1 and 2 (the OS saves XMM and YMM state); CPUID leaf 7 EBX bit 5 (AVX2).
- `Dot` and `L2Sq` use assembly only when `useAVX2 && len >= 16`; shorter inputs use Go.
- On non-amd64 builds `useAVX2` is the constant `false`, and the asm symbols are stubs that
  panic (`kernel_other.go`); they are never called.

### 5.3 Assembly contract (`kernel_amd64.s`)
- ABI0, `NOSPLIT`, zero local frame. Argument frame sizes: `dotAVX2`/`l2sqAVX2` `$0-28`,
  `dotBatchAVX2`/`dotBatch8AVX2` `$0-40`, `cpuid` `$0-24`, `xgetbv` `$0-8`. Go declarations carry
  `//go:noescape`.
- Allowed registers: AX, BX, CX, DX, SI, DI, R8–R13, X/Y0–Y15. **Never touch R14 (g) or R15.**
- Every AVX function ends with `VZEROUPPER` before `RET`.
- `dotAVX2`/`l2sqAVX2`: 32 floats per iteration into 4 independent accumulators, then an
  8-wide loop, then a horizontal reduce, **then** a scalar tail. The tail must come after the reduce,
  because scalar VEX ops zero the upper lanes of their destination register.
- `dotBatchAVX2`: 16 floats per inner iteration, 2 accumulators, no tail (stride % 16 == 0).
- `dotBatch8AVX2`: one row chunk is loaded into Y8 and fused-multiply-added against 8 queries
  (accumulators Y0–Y7). Queries are addressed from one base register with scaled offsets
  (R10 = 1×, R11 = 3×, R12 = 5×, R13 = 7× stride bytes). No tail (stride % 8 == 0).
- Go assembler operand order is reversed from Intel: `VFMADD231PS m, a, acc` means `acc += a*m`.

### 5.4 Numerics
Results are compared with a float64 reference, not by exact equality. Tolerance:
`|got - want| <= 1e-5*n + 1e-4*|want|`.

### 5.5 Tests [DONE]
- `TestDotAndL2AllLengths`: every length 0–200 (all three code paths), plus 384, 768, 1024, 1536, 3072.
- `TestDotBatch`: strides 16, 32, 48, 784 × 37 rows.
- `TestDotBatch8`: strides 8, 16, 64, 784 × 29 rows × 8 queries.
- `TestDetect`: logs `Accelerated()`.
- `go vet ./...` must also pass with `GOARCH=arm64`.

## 6. Package `search`

### 6.1 Index [DONE]

```go
type Result struct { ID uint32; Score float32 } // Score = dot product, higher is closer
type Index struct {
    TileRows int // rows per cache tile; default max(16, 1 MiB / (stride*4))
    Workers  int // goroutines; default runtime.GOMAXPROCS(0)
    Live     func(id uint32) bool // optional; rows where it returns false never appear in results
}
func New(rows []float32, stride, n int) *Index
func (ix *Index) SearchBatch(queries [][]float32, k int) [][]Result
```

Preconditions: `stride % 8 == 0`; `len(rows) >= n*stride`; rows are zero-padded; every
`len(query) <= stride`.

`SearchBatch` algorithm:
1. If `len(queries) == 0` or `k <= 0`, return `make([][]Result, len(queries))`.
2. **Pack** the queries into groups of 8, each `stride` floats, contiguous and zero-padded. Unused
   slots in the last group stay zero.
3. **Tile** the rows into `tiles = ceil(n / TileRows)` tiles. Use `min(Workers, tiles)` workers
   (at least 1). Worker `w` owns the contiguous tile range `[w*tiles/W, (w+1)*tiles/W)`.
4. For each tile, **every** query group runs `kernel.DotBatch8` over it before the worker moves
   to the next tile, so the tile is read from RAM once and then re-read from L2.
5. Each worker keeps one top-k min-heap per query. When the heap is full, a score `<=` the root is
   rejected with a single comparison. Only a score that would enter the heap is checked against
   `Live` (if set), so the filter costs nothing for most rows.
6. Merge the per-worker heaps into a final top-k per query, sorted by descending score.
   Order among equal scores is unspecified.

Output: per query, `min(k, live rows)` results.

**Tombstones [DONE]:** to search an arena with deletes, set `Live = arena.IsLive` and hold an
arena `Guard` around `SearchBatch` so no row is reused mid-scan.

### 6.2 Batcher [DONE]

```go
func NewBatcher(ix *Index, maxBatch int, maxWait time.Duration) *Batcher
func (b *Batcher) Search(ctx context.Context, q []float32, k int) ([]Result, error)
func (b *Batcher) Close()
var ErrClosed = errors.New("search: batcher closed")
```

- The request channel is buffered to `maxBatch*4`. One dispatcher goroutine runs batches one at a time.
- The dispatcher blocks until the first request of a batch arrives, then starts a `maxWait` timer.
  It dispatches as soon as the batch holds `maxBatch` requests or the timer fires.
- The batch runs with `k = max(k_i)`, and each result is trimmed to its own `k_i`.
- `Search` returns `ctx.Err()` if the context ends while enqueueing or waiting. A request that was
  already enqueued still runs, and its reply is discarded (the reply channel is buffered to 1).
- Every enqueued request gets exactly one reply: results, or `ErrClosed`.
- **Shutdown [DONE] (fixed B-02 and B-03):**
  - `Close` is idempotent (`sync.Once`). It closes `done`, then takes `mu` for writing and closes
    the request channel. `Search` holds `mu` for reading while it sends, and fails with
    `ErrClosed` if `done` is closed, so no send can reach a closed channel.
  - After `done` is closed, the dispatcher stops filling batches. Requests not yet dispatched,
    and every request it drains afterwards, get `ErrClosed`. It exits when the request channel is
    closed. A batch already running when `Close` is called completes normally.

### 6.3 Tests [DONE]
- `TestSearchBatchMatchesBruteForce`: n = 5000, dim 100, stride 112, TileRows = 333 (uneven
  tiles), k = 10, query counts 1, 7, 8, 9, 20. Result ids must match an exact float64 top-k in order.
- `TestBatcherGroupsConcurrentRequests`: 40 concurrent `Search` calls, maxBatch 16, maxWait 5 ms;
  every result matches brute force. Must pass `-race`.
- `TestBatcherCloseRace`: 1000 concurrent `Search` calls race `Close`; every call returns (a
  result or `ErrClosed`) within 1 s; a second `Close` doesn't panic; `Search` after `Close`
  returns `ErrClosed`.
- `TestDeletedRowsNeverReturned` (linux): delete the 100 best matches for a query from an arena;
  with `Live = arena.IsLive`, none is returned and results equal brute force over the live rows.
- `BenchmarkThroughput`: 100k × 768, k = 10, batch sizes 1, 8, 32, 64; reports `queries/s` and `ms/batch`.

## 7. Package `spsc`

Ring buffers for exactly **one producer goroutine and one consumer goroutine**. Four versions
exist for teaching and benchmarking; production code uses `Ring[T]`.

### 7.1 Shared rules
- Capacity is rounded up to a power of two (`roundPow2`; a capacity `<= 1` becomes 1).
  `mask = cap - 1`, and the slot for counter `i` is `i & mask`.
- `head` and `tail` are monotonic `uint64` counters. Empty: `tail == head`. Full: `tail - head == cap`.
- **Ordering rule (lock-free versions):** the producer writes the slot, *then* does `tail.Store`.
  The consumer does `tail.Load`, reads the slot, clears it (`zero` / `clear`), *then* does
  `head.Store`. Every cross-goroutine index access uses `sync/atomic`.
- Popped slots are zeroed so the ring doesn't keep items reachable for the GC.
- Calling producer methods from more than one goroutine (or consumer methods from more than one)
  is undefined behaviour. It is not detected.

### 7.2 Versions

| Type | Constructor | Notes |
|---|---|---|
| `MutexRing[T]` | `NewMutexRing[T](capacity)` | one `sync.Mutex`; the only version safe with multiple producers or consumers |
| `NaiveRing[T]` | `NewNaiveRing[T](capacity)` | atomic head and tail, adjacent in memory (false sharing, on purpose) |
| `PaddedRing[T]` | `NewPaddedRing[T](capacity)` | 64-byte padding before `head`, between `head` and `tail`, and after `tail` |
| `Ring[T]` | `New[T](capacity)` | padding plus cached indices plus batch operations (production type) |

### 7.3 `Ring[T]` [DONE]

Layout, in 64-byte lines: `[pad] [head, cachedTail, pad] [tail, cachedHead, pad] [buf, mask, pad]`.
`TestLayout` asserts that `tail`'s offset is at least 64 bytes beyond `head`'s. The reference
offsets are head at 64 and tail at 128.

| Method | Caller | Semantics |
|---|---|---|
| `Push(v T) bool` | producer | If `tail - cachedHead == cap`, refresh `cachedHead = head.Load()`; if still full, return false. Otherwise write the slot, `tail.Store(t+1)`, return true. |
| `Pop() (T, bool)` | consumer | If `head == cachedTail`, refresh `cachedTail = tail.Load()`; if still empty, return the zero value and false. Otherwise read the slot, zero it, `head.Store(h+1)`. |
| `PushBatch(vs []T) int` | producer | Refresh `cachedHead` only if the cached free space is `< len(vs)`. Push `n = min(len(vs), free)` items as at most two `copy` segments (wrap-around). One `tail.Store(t+n)` if `n > 0`. Returns `n`. |
| `PopBatch(dst []T) int` | consumer | Refresh `cachedTail` only if the cached count is `< len(dst)`. Pop `n = min(len(dst), avail)` as at most two segments, `clear` each, then one `head.Store(h+n)`. Returns `n`. |
| `Cap() int` | any | capacity after rounding |

### 7.4 Tests [DONE]
- `TestFIFOAllVariants`: 200k values, producer goroutine to consumer, exactly once and in order,
  for all four types. Must pass `-race`.
- `TestFullAndEmpty`: capacity 4; pop from empty fails; the 5th push fails; FIFO order;
  1000 wrap-arounds.
- `TestBatchOrder`: 300k values, producer batches of 37 (odd, to force partial batches),
  consumer batches of 64, order preserved.
- `TestLayout`: the padding assertion above.
- Benchmarks: `Channel`, `Mutex`, `Naive`, `Padded`, `Cached`, `CachedBatch64` (yield with
  `runtime.Gosched` when blocked); `SpinMutex`, `SpinNaive`, `SpinPadded`, `SpinCached` (pure spin);
  `ChannelBatch64`.

## 8. Reference performance

Measured on a 2-vCPU Intel Xeon cloud VM (AVX2, FMA, AVX-512F, VNNI), Go 1.24.7, `GOMAXPROCS=2`.
Treat these as **regression baselines**, not targets. They are noisy. Re-measure on the target
machine, report the median of at least 5 runs, and compare relative changes.

| Benchmark | Result |
|---|---|
| `kernel.Dot`, 768 dims, both vectors in cache | 58 ns (plain Go: 492 ns) |
| `kernel.L2Sq`, 768 dims | 68 ns (plain Go: 507 ns) |
| `kernel.DotBatch`, 100k × 768 (307 MB) | 31 ms (per-row Go loop: 61 ms) |
| `search.SearchBatch`, 100k × 768, k = 10 | batch 1: 43 q/s · batch 8: 322 q/s · batch 32: 509 q/s · batch 64: 562 q/s |
| `spsc`, ns per item (median of 7, yield when blocked) | Channel 62.1 · Mutex 65.7 · Naive 150.6 · Padded 200.6 · Cached 111.4 · Cached, batches of 64: 4.66 · Channel of 64-item arrays: 1.69 |

How to read the SPSC numbers: sending one item at a time, the lock-free rings did not beat
channels on this VM, most likely because the queue sat near empty. Batching dominates. Before
optimising SPSC further, re-benchmark on dedicated cores with threads pinned (backlog `T-12`).

## 9. Backlog

Work these in order. Each item is done only when its acceptance criteria pass along with the
whole test suite (section 10).

| ID | Priority | Task | Acceptance criteria |
|---|---|---|---|
| T-01 | P0 [DONE] | Fix B-01 (torn read on slot reuse). Hold freed ids in a *pending* list and move them to `free` only after every reader active at delete time has finished. Use epoch-based reclamation: readers enter and exit an epoch around `Range`/`Block` scans. | New test: readers verify a per-row checksum while a writer deletes and re-adds rows in a loop. Zero mismatches in 10 s under `-race`. |
| T-02 | P0 [DONE] | Tombstones in search. `search.New` accepts an optional `Live func(id uint32) bool`, or a bitset view exported by `Arena`. Dead rows never appear in results. | Test: delete the 100 best matches for a query; none of them are returned. |
| T-03 | P0 [DONE] | Fix B-02/B-03 in `Batcher`: `Close` is idempotent (`sync.Once`); after `Close`, pending and in-flight enqueues get `ErrClosed`. The dispatcher drains the channel on shutdown and replies `ErrClosed` to each request. | Test: 1000 concurrent `Search` calls racing `Close`; every call returns (a result or `ErrClosed`) within 1 s; calling `Close` twice doesn't panic. |
| T-04 | P1 | Non-Linux build: `arena_other.go` using `syscall.Mmap` without `MAP_NORESERVE` on darwin; the huge-page option is a no-op there. | `GOOS=darwin go vet ./...` passes. |
| T-05 | P1 | External ids: map `string` or `uint64` ids to internal `uint32` ids and back; handle upsert and delete. | Round-trip test; memory use per id documented. |
| T-06 | P1 | `kernel.Normalize(v []float32)` (in place, L2), and an `Options.Normalize` flag on insert for cosine similarity. | Cosine top-k matches a float64 reference. |
| T-07 | P1 | Metadata filters: a column store plus Roaring-style bitmaps. `SearchBatch` takes a per-query filter. Planner: if a filter matches < 5% of rows, brute-force only those rows; otherwise filter at heap push. | Results equal filtered brute force; benchmark at 1%, 10% and 50% selectivity. |
| T-08 | P2 | int8 quantised scan with rerank: store int8 codes plus scale; AVX2 (`VPMADDUBSW`/`VPMADDWD`) and VNNI (`VPDPBUSD`) kernels, dispatched by CPUID; take the top `k*4` by int8 score, rerank with float32. | recall@10 >= 0.99 against exact search on random and normalised data; scan faster than float32 on the 100k × 768 benchmark. |
| T-09 | P2 | AVX-512 `DotBatch16` (16 queries per pass) with CPUID/XCR0 detection (bits 5–7 for the ZMM state). | Same tests as `DotBatch8`; throughput benchmark at batch 16 and 64. |
| T-10 | P2 | Snapshots: a file-backed `mmap` arena plus an append-only log; recover on start. | Kill-and-restart test: every acknowledged insert survives. |
| T-11 | P2 | arm64 NEON kernels (`FMLA`) for `Dot`, `L2Sq`, `DotBatch`, `DotBatch8`. | Kernel tests pass on arm64 hardware; benchmarks recorded. |
| T-12 | P3 | Benchmark harness with pinned threads (`runtime.LockOSThread` plus `SchedSetaffinity`) and an occupancy histogram for the SPSC variants. | A report of median and spread over 7 runs, pinned vs unpinned. |
| T-13 | P3 | Pipeline: `Batcher` → per-worker `spsc.Ring` → workers → per-worker result rings → merger, replacing goroutine-per-tile. | Same results as `SearchBatch`; throughput and p99 latency compared. |
| T-14 | P3 | NUMA: one arena partition per socket, with each worker running on the socket that owns its partition. | Benchmark on a two-socket machine. |

## 10. Definition of done (every change)

Run from the repository root; all must pass:

```sh
gofmt -l .                      # prints nothing
go vet ./...
GOARCH=arm64 go vet ./...       # fallbacks still compile
go test -race -count=1 ./...
```

For performance changes, also run the affected benchmarks before and after (median of 5 or more)
and record the numbers in the pull request or commit message:

```sh
go test -run x -bench . -count 5 ./kernel ./search ./spsc
```

## 11. Conventions

- **Panics vs errors:** programmer errors (wrong lengths, a bad stride) panic with a
  `"<pkg>: <reason>"` message. Runtime conditions (full, closed, mmap failure) return `error`.
- **Hot paths allocate nothing:** kernels, `Range`, `Push`/`Pop`. Check with `-benchmem`.
- **Assembly:** every asm function has a Go declaration with `//go:noescape`, a portable fallback,
  and a `!amd64` stub. It is tested against a float64 reference at lengths that hit every loop tail.
- **Comments:** each package opens with a doc comment explaining the design. Each non-obvious
  ordering (publish after write, clear before release) has a one-line comment saying why.
- **Unsafe:** only in `vecarena` (mmap views). No Go pointers are ever written into mmap'd memory.
