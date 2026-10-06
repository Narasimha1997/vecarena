# vecarena

vecarena is an in-memory vector search engine for Go. It does exact top-k search over float32
vectors and makes the brute force scan as fast as the hardware allows. On a laptop CPU it searches
100,000 vectors of 768 dimensions at about 850 queries per second.

**Features**

* **Exact results.** Every query is compared against every vector, so recall is always 100%.
  There is no index to build or tune.
* **Off heap storage.** Vectors live in `mmap` memory that the Go garbage collector never scans.
* **SIMD kernels.** Hand written AVX2 and FMA assembly, about 8x faster than plain Go, with
  automatic fallback on other CPUs.
* **Batched, multi core search.** Eight queries share each pass over memory, spread across all
  cores with cache tiling.
* **Lock free reads.** Searches run alongside inserts and deletes without taking a lock, and
  never see a half written row.
* **Request batcher.** Groups single queries from many goroutines into batches automatically.
* **Instant updates.** Inserts and deletes take effect immediately, and deleted slots are reused.
* **Your own ids.** Store vectors under string or integer keys, with upsert and delete.
* **Metadata filters.** Filter by tags and number ranges using compressed bitmaps. Very selective
  filters only score the matching rows, so they get faster, not slower.
* **Cosine similarity.** Optional normalisation on insert turns dot product into cosine.
* **Zero dependencies.** Pure Go plus a little assembly, standard library only.

## Why brute force?

Approximate indexes like HNSW are the right tool once you have tens of millions of vectors. Below
that, a well tuned flat scan is often good enough, and it comes with real advantages:

* Results are exact, so there is no recall tuning and nothing to rebuild.
* Inserts and deletes are cheap and take effect immediately.
* Memory use is predictable: one row per vector, no graph on top.
* The code is small enough to read in an afternoon.

vecarena focuses on making that scan bandwidth bound rather than compute bound.

## How it works

**Off heap storage.** Vectors live in a single anonymous `mmap` region outside the Go heap, so the
garbage collector never scans them. Each row is padded to a multiple of 64 bytes, so every row
starts on its own cache line. Memory is committed lazily as rows are written, and on Linux huge
pages can be requested with `madvise`.

**SIMD kernels.** Dot product and squared L2 distance use hand written AVX2 and FMA assembly,
detected at startup through CPUID. Other CPUs fall back to portable Go automatically.

**Query batching.** `DotBatch8` scores eight queries against each row while the row is still in
registers. Rows are read from memory once per batch instead of once per query, which is where
most of the throughput at larger batch sizes comes from.

**Cache tiling and parallelism.** Rows are split into tiles of roughly 1 MiB. Each worker
goroutine owns a range of tiles, runs every query group over a tile while it sits in L2, and keeps
its own top-k heaps. The heaps are merged at the end.

**Safe concurrent updates.** One writer and many readers can work at the same time without
readers taking a lock. Deleted slots are reused only after every reader that might still see them
has finished, using epoch based reclamation, so a scan never reads a half written row.

**Request batching.** `search.Batcher` collects single queries from many goroutines into one
batch, so a server handling one query per request still gets the batched throughput.

**Filter planning.** Metadata lives in a column store, and every filter turns into a Roaring
style compressed bitmap of matching rows. If a filter matches fewer than 5% of rows, the search
scores only those rows. Otherwise it runs the normal batched scan and checks the filter only for
rows good enough to enter the top k, which costs almost nothing.

## Requirements

* Go 1.24 or newer
* Linux or macOS for off heap `mmap` storage. On Windows and other systems vectors are kept in a
  plain Go byte slice instead. The garbage collector still never scans it, but the memory is
  committed up front rather than as rows are written.
* For the fast path, an x86 CPU with AVX2 and FMA (Intel Haswell or newer, AMD Zen or newer).
  Everything also builds and runs on arm64 using the portable Go code, just more slowly.

## Installation

The module path is currently `vecarena`, so the simplest way to use it is to clone the
repository next to your project and point your `go.mod` at it:

```sh
git clone https://github.com/Narasimha1997/vecarena.git
```

Then, in your project's `go.mod`:

```
require vecarena v0.0.0
replace vecarena => ../vecarena
```

To check that everything works on your machine:

```sh
cd vecarena
go test ./...
```

## Usage

### Storing vectors

```go
import "vecarena"

// Reserve space for up to a million 768 dimensional vectors.
// RAM is committed only as rows are written.
arena, err := vecarena.New(768, 1_000_000, vecarena.Options{HugePages: true})
if err != nil {
	log.Fatal(err)
}
defer arena.Close()

id, err := arena.Add(embedding) // embedding is a []float32 of length 768
row := arena.Get(id)            // a view into arena memory, not a copy
arena.Delete(id)                // the slot is reused by a later Add
```

`Add` returns a `uint32` id. When you delete a row, its id goes back into a free list and a later
`Add` reuses it. `Get` returns a slice that points directly into the arena, so treat it as read
only.

### Using your own keys

Most applications have their own ids, like document names or database keys. A `Collection` maps
those to arena rows for you:

```go
docs, err := vecarena.NewCollection[string](768, 1_000_000, vecarena.Options{})
if err != nil {
	log.Fatal(err)
}
defer docs.Close()

id, err := docs.Upsert("doc-42", embedding) // insert, or replace if the key exists
docs.Delete("doc-42")

key, ok := docs.Key(id) // turn a search result back into your key
```

The key can be any comparable type, such as `string` or `uint64`. Replacing a key writes the new
vector to a fresh row and then deletes the old one, so a concurrent search never sees a half
updated vector. Because of that, the row id for a key can change on every upsert. The mapping
costs about 46 bytes per `uint64` key and 90 bytes per 16 byte string key, on top of the vectors.

Search the collection through `docs.Arena()`, exactly as shown below.

### Cosine similarity

Set `Normalize` when you create the arena or collection, and normalise each query:

```go
docs, err := vecarena.NewCollection[string](768, 1_000_000, vecarena.Options{Normalize: true})
// ...
kernel.Normalize(query)
```

Every stored vector is scaled to unit length on insert (your slice is not modified), so the dot
product scores that search returns are cosine similarities.

### Searching

```go
import "vecarena/search"

n := 10_000 // number of rows written so far
ix := search.New(arena.Block(0, n), arena.Stride(), n)
ix.Live = arena.IsLive // skip deleted rows

guard := arena.Enter()
results := ix.SearchBatch(queries, 10) // queries is a [][]float32
guard.Exit()

for i, hits := range results {
	for _, h := range hits {
		fmt.Printf("query %d: id=%d score=%.3f\n", i, h.ID, h.Score)
	}
}
```

Results come back sorted by score, highest first. The score is the dot product, so higher means
more similar.

A few things to keep in mind:

* Pass many queries to `SearchBatch` at once when you can. Batching is the biggest single
  performance lever; see the benchmarks below.
* `Enter` and `Exit` mark a read section. While it is open, no row you might be reading will be
  overwritten. Keep it around the search and nothing else, because deleted slots cannot be reused
  until it closes.
* An `Index` covers the rows that existed when it was created. Creating one is cheap (it is just
  a small struct), so build a new one after adding rows.
* `Workers` and `TileRows` on the index can be changed if the defaults (all cores, about 1 MiB
  tiles) do not suit your machine.

### Filtering by metadata

Store metadata per row in a `filter.Columns`, then pass one filter per query to
`SearchFiltered`:

```go
import "vecarena/filter"

meta := filter.NewColumns()
meta.SetString(id, "lang", "en")
meta.SetNumber(id, "year", 2024)

recent := meta.Select(filter.And(
	filter.Eq("lang", "en"),
	filter.AtLeast("year", 2020),
))

guard := arena.Enter()
results := ix.SearchFiltered(queries, 10, []search.Filter{recent})
guard.Exit()
```

Available expressions are `Eq`, `In`, `Range`, `AtLeast`, `AtMost`, `Has`, `And`, `Or` and `Not`.
`Select` returns a bitmap, so you can build a filter once and reuse it across many searches. Use
`nil` in the filter list for a query that should not be filtered.

Metadata is stored by row id, so when you delete a row, or upsert a key in a collection (which
moves it to a new row), call `meta.Clear` on the old row id and set the values on the new one.

### Serving single queries

When queries arrive one at a time, for example one per HTTP request, put a `Batcher` in front of
the index:

```go
b := search.NewBatcher(ix, 64, time.Millisecond) // batches of up to 64, wait at most 1 ms
defer b.Close()

hits, err := b.Search(ctx, query, 10)
```

The batcher waits until it has 64 queries or 1 ms has passed, whichever comes first, then runs
them together. `Search` respects context cancellation, and after `Close` it returns
`search.ErrClosed`.

### Distance kernels

The kernels are usable on their own:

```go
import "vecarena/kernel"

s := kernel.Dot(a, b)  // dot product
d := kernel.L2Sq(a, b) // squared Euclidean distance
kernel.Accelerated()   // true when the AVX2 code path is active
```

## Benchmarks

All numbers below were measured on a laptop with an Intel Core i7 9750H (6 cores, 12 threads,
AVX2 and FMA), 32 GB of RAM, Linux and Go 1.24.7. Each number is the median of 5 runs. Treat them
as a rough guide: results on your hardware will differ, and laptop numbers are noisy.

### End to end search

100,000 vectors with 768 dimensions (293 MiB), top 10, all 12 threads:

| Batch size | Queries per second | Latency per batch (p50) | Latency per batch (p99) |
|---:|---:|---:|---:|
| 1 | 86 | 11.2 ms | 22.2 ms |
| 8 | 641 | 12.1 ms | 19.2 ms |
| 32 | 831 | 36.3 ms | 64.9 ms |
| 64 | 848 | 72.4 ms | 114.0 ms |

Going from one query to eight multiplies throughput by about 7.5 for almost no extra latency,
because the scan is limited by memory bandwidth and eight queries share one pass over the data.
Past that point the gains flatten out as the scan becomes compute bound.

With the `Batcher` in front and 64 goroutines each sending one query at a time, throughput was
839 queries per second, with a p50 latency of 73 ms and p99 of 110 ms. That matches the batch of
64 row above, so the batcher adds very little overhead.

Recall@10 against a separate brute force check was 1.0, as expected for an exact search.

### Filtered search

Same data, batches of 8 queries, each with its own random filter:

| Rows matching the filter | Queries per second |
|---:|---:|
| 1% | 9,425 |
| 10% | 624 |
| 50% | 625 |
| No filter | 643 |

At 1% the planner scores only the matching rows, which is about 15 times faster than a full
scan. From 5% upwards the normal scan runs, and the filter adds no measurable cost.

### Kernels

| Operation | AVX2 | Plain Go | Speedup |
|---|---:|---:|---:|
| Dot product, 768 dims | 75 ns | 580 ns | 7.8x |
| Squared L2, 768 dims | 70 ns | 614 ns | 8.8x |
| One query against 100k x 768 rows, single thread | 16.3 ms | 87.0 ms | 5.3x |

The single thread full scan reads about 19 GB/s, which is close to what one core can pull from
memory on this machine.

### Ring buffers

The `spsc` package contains several single producer, single consumer ring buffers that are used
for experiments. Nanoseconds per item:

| Queue | One item at a time | Batches of 64 |
|---|---:|---:|
| Go channel | 74.0 | 1.8 |
| Mutex ring | 68.1 | |
| Lock free ring, no padding | 81.0 | |
| Lock free ring, padded | 77.8 | |
| Lock free ring, padded with cached indices | 24.0 | 2.3 |

The main lesson is that batching matters far more than which queue you pick.

### Running the benchmarks yourself

The `vecbench` command runs the full end to end benchmark:

```sh
go run ./cmd/vecbench
```

It loads the arena, measures raw scan bandwidth, search throughput and latency for several batch
sizes, batcher latency under concurrent clients, and finally checks recall. Run
`go run ./cmd/vecbench -h` for all options. Some useful ones:

```sh
# a smaller, quicker run
go run ./cmd/vecbench -n 20000 -dim 256 -duration 500ms -runs 3

# a larger dataset with different batch sizes
go run ./cmd/vecbench -n 1000000 -dim 384 -batches 1,16,64,128

# limit the number of search workers
go run ./cmd/vecbench -workers 4
```

The per package micro benchmarks use the standard Go tooling:

```sh
go test -run x -bench . -count 5 ./kernel ./search ./spsc

# just the filtered search benchmark
go test -run x -bench Filtered -count 5 ./search
```

## Project layout

```
vecarena/
├── arena.go           off heap vector storage
├── mem_*.go           memory mapping for Linux, macOS and everything else
├── collection.go      your own keys mapped to arena rows
├── kernel/            dot product and L2 kernels, AVX2 assembly and Go fallbacks
├── search/            batched and filtered top-k search, and the request batcher
├── filter/            compressed bitmaps, metadata columns and filter expressions
├── spsc/              single producer, single consumer ring buffers
├── cmd/vecbench/      end to end benchmark tool
└── SPEC.md            detailed design, invariants and roadmap
```

[SPEC.md](SPEC.md) is the full specification: every public API, the concurrency rules, the
assembly conventions and the backlog. If you want to understand why something works the way it
does, start there.

## Roadmap

Planned work, roughly in order:

* int8 quantised scans with float32 rescoring
* AVX-512 kernels that score 16 queries per pass
* Snapshots to disk and recovery on restart
* ARM NEON kernels
* NUMA aware partitioning for multi socket machines

The backlog in [SPEC.md](SPEC.md) has the details and acceptance criteria for each item.

## Current limitations

* Vectors are float32 and the similarity is the dot product (or cosine, with normalisation).
* Metadata is not tied to collections yet, so you keep it in step yourself when rows move.
* Capacity is fixed when the arena is created.
* Everything is in memory. Nothing is persisted yet.
* There is one writer at a time. Concurrent writers are serialised by a mutex.

## Contributing

Contributions are welcome. Before opening a pull request, please make sure all of these pass from
the repository root:

```sh
gofmt -l .                     # should print nothing
go vet ./...
GOARCH=arm64 go vet ./...      # the portable fallbacks must still compile
GOOS=darwin go vet ./...       # the macOS build
GOOS=windows go vet ./...      # the fallback memory build
go test -race -count=1 ./...
```

The full test suite takes about 20 seconds, mostly because one test stress tests concurrent
deletes for 10 seconds and another measures memory for a million keys. Use `go test -short ./...` for a quicker loop while you work.

If your change affects performance, please run the relevant benchmarks before and after (at least
5 runs each) and include both sets of numbers in the pull request.

## License

vecarena is released under the GNU General Public License v3.0. See [LICENSE](LICENSE) for the
full text.
