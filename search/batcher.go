package search

import (
	"context"
	"errors"
	"sync"
	"time"
)

type Batcher struct {
	ix       *Index
	maxBatch int
	maxWait  time.Duration
	reqs     chan request
	done     chan struct{}

	// mu is held for reading while a Search sends on reqs, and for writing
	// while Close closes reqs, so no send can hit a closed channel.
	mu        sync.RWMutex
	closeOnce sync.Once
}

type request struct {
	q     []float32
	k     int
	reply chan response
}

type response struct {
	res []Result
	err error
}

var ErrClosed = errors.New("search: batcher closed")

func NewBatcher(ix *Index, maxBatch int, maxWait time.Duration) *Batcher {
	b := &Batcher{
		ix: ix, maxBatch: maxBatch, maxWait: maxWait,
		reqs: make(chan request, maxBatch*4),
		done: make(chan struct{}),
	}
	go b.loop()
	return b
}

// Search queues q and waits for its batch. Once enqueued, a request always
// gets a reply (results, or ErrClosed if the batcher closed first).
func (b *Batcher) Search(ctx context.Context, q []float32, k int) ([]Result, error) {
	reply := make(chan response, 1)
	if err := b.enqueue(ctx, request{q, k, reply}); err != nil {
		return nil, err
	}
	select {
	case r := <-reply:
		return r.res, r.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (b *Batcher) enqueue(ctx context.Context, r request) error {
	b.mu.RLock()
	defer b.mu.RUnlock()
	select {
	case <-b.done:
		return ErrClosed
	default:
	}
	select {
	case b.reqs <- r:
		return nil
	case <-b.done:
		return ErrClosed
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Close stops the batcher. It is idempotent. Requests not yet dispatched get
// ErrClosed; a batch already running completes normally.
func (b *Batcher) Close() {
	b.closeOnce.Do(func() {
		// Closing done first unblocks Searches waiting to send, so the
		// write lock below cannot wait on a full channel.
		close(b.done)
		b.mu.Lock()
		close(b.reqs)
		b.mu.Unlock()
	})
}

func (b *Batcher) loop() {
	batch := make([]request, 0, b.maxBatch)
	timer := time.NewTimer(time.Hour)
	timer.Stop()
	// After Close, the loop keeps draining reqs, replying ErrClosed, until
	// Close closes the channel.
	for {
		r, ok := <-b.reqs
		if !ok {
			return
		}
		batch = append(batch, r)
		timer.Reset(b.maxWait)

	fill:
		for len(batch) < b.maxBatch {
			select {
			case r, ok := <-b.reqs:
				if !ok {
					break fill
				}
				batch = append(batch, r)
			case <-timer.C:
				break fill
			case <-b.done:
				break fill
			}
		}
		timer.Stop()

		select {
		case <-b.done:
			for _, r := range batch {
				r.reply <- response{err: ErrClosed}
			}
		default:
			b.run(batch)
		}
		batch = batch[:0]
	}
}

func (b *Batcher) run(batch []request) {

	k := 0
	qs := make([][]float32, len(batch))
	for i, r := range batch {
		qs[i] = r.q
		k = max(k, r.k)
	}
	res := b.ix.SearchBatch(qs, k)
	for i, r := range batch {
		out := res[i]
		if len(out) > r.k {
			out = out[:r.k]
		}
		r.reply <- response{res: out}
	}
}
