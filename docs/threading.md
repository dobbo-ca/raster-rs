# Threading

Every kernel splits its destination rows into bands, one per thread:

- `scale2to1` and `luma`: bands of whole destination rows.
- `rotate90`: bands of whole 16-row tile bands of the destination, so tiles are never cut.
- `resample`: the horizontal pass over rows of the intermediate plane, the vertical pass over
  rows of the destination; the vertical bands need an accumulator row each (see `max_alloc` in limits.md).

A band has at least about 32 us of work (see limits.md), so small images never leave the calling thread and
never build the shared pool. Bands write disjoint rows, so the output does not depend on how the
rows are split.
