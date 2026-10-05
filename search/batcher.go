package search

import (
	"context"
	"errors"
	"time"
)

type Batcher struct {
	ix       *Index
	maxBatch int
	maxWait  time.Duration
	reqs     chan request
	done     chan struct{}
}

type request struct {
	q     []float32
	k     int
	reply chan []Result
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

func (b *Batcher) Search(ctx context.Context, q []float32, k int) ([]Result, error) {
	reply := make(chan []Result, 1)
	select {
	case b.reqs <- request{q, k, reply}:
	case <-b.done:
		return nil, ErrClosed
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	select {
	case res := <-reply:
		return res, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (b *Batcher) Close() { close(b.done) }

func (b *Batcher) loop() {
	batch := make([]request, 0, b.maxBatch)
	timer := time.NewTimer(time.Hour)
	timer.Stop()
	for {

		select {
		case r := <-b.reqs:
			batch = append(batch, r)
		case <-b.done:
			return
		}
		timer.Reset(b.maxWait)

	fill:
		for len(batch) < b.maxBatch {
			select {
			case r := <-b.reqs:
				batch = append(batch, r)
			case <-timer.C:
				break fill
			}
		}
		timer.Stop()

		b.run(batch)
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
		r.reply <- out
	}
}
