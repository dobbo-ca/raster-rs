# Resource limits

Every function takes a `Limits` and a cancel flag; there is no unlimited call.

| Field | Unit | Default | Over it |
|---|---|---|---|
| `max_pixels` | pixels (the larger of source and destination) | `1 << 26` | `Error::PixelLimit` |
| `max_alloc` | bytes | `1 << 28` (256 MiB) | `Error::AllocLimit` |
| `max_output` | bytes written to the destination | `1 << 28` | `Error::OutputLimit` |
| `max_threads` | threads, the calling one included | `std::thread::available_parallelism()`, 1 if unknown | none: the call uses fewer threads |
| `pool` | `Option<ThreadPool>` | `None`: the crate's shared pool | none |

- `max_alloc` is the enforced peak working memory of one call, all its threads together. Only
  `resample` allocates: the tap tables, the weight row, the intermediate plane and one `i32`
  accumulator row for each band of the vertical pass running at once. Each is charged before it
  is allocated and released when it is freed. A `Resampler` keeps these buffers between calls
  (`docs/performance.md`): each call charges the kept buffers at their capacity, and what it must
  grow, before anything grows; if the kept buffers would take the call over `max_alloc`, the ones
  bigger than the call needs are freed first, so a call that passes as the free `resample` passes
  on any `Resampler`, and after `Error::AllocLimit` it holds nothing. Go's up-front scratch check
  also stays: a call whose single-thread need is over `max_alloc` is refused with
  `Error::AllocLimit` before anything is allocated. Beyond that need, extra bands are only started while their accumulator
  rows fit, so a call that fits at 1 thread succeeds at every `max_threads`. The caller's own
  buffers are not charged. The crate calls no other port, so there are no nested calls to share
  the budget with. `scale2to1`, `luma` and `rotate90` allocate nothing, and a second
  `Resampler::resample` of the same shape allocates nothing at `max_threads` 1, if its `max_alloc` is
  no tighter and its `max_threads` no lower than in the call that built the buffers (else it trims them and
  allocates again).
- `max_threads: 1` runs everything on the calling thread, spawns nothing and never touches a
  pool (0 is treated as 1). Above 1, each kernel splits its rows into bands of at least about
  32 us of work each (32K pixels, or about 500K pixels on the SIMD luma, grey `scale2to1`
  and grey AVX-512 `rotate90` tile paths); a smaller image stays on the calling thread. The output is byte for
  byte the same for every `max_threads` and every pool.
- The other threads of a call are the workers of a `ThreadPool`, started once and kept, so
  handing them a task costs microseconds. A call takes at most `max_threads - 1` workers besides
  the calling thread, and no more than the pool has.
  - `Limits::pool` is a pool the caller made with `ThreadPool::new(threads)` and keeps; it always
    wins. Clones of a `ThreadPool` share the threads.
  - Without one, a call with `max_threads` above 1 uses the crate's shared pool: a `OnceLock`
    inside the crate, built on the first call that has work for a second thread, with
    `available_parallelism` threads and never more (it never grows), and holding no other state.
    A call with `max_threads` 1, or with a caller's pool, never builds or touches it. If the
    shared pool cannot be built, calls without a pool of their own run on the calling thread.
  - A panic on a pool thread propagates to the caller (rayon re-raises it).
- `cancel: &AtomicBool` is checked per row (per 16-row tile band for `rotate90`) by every thread
  of the call. When it is set the call returns `Error::Cancelled`, and `dst` can be partly
  written: only rows finished before the flag was seen, and which rows that is depends on
  `max_threads`. A call that returns `Ok` has the same bytes whatever the thread count.
- `Error::OutOfMemory` is the allocator refusing a size that passed the limits (or, from
  `ThreadPool::new`, the OS refusing to start the threads).

## Errors

One `Error` enum, one variant per Go error: `Invalid` is `ErrInvalid`; `PixelLimit`,
`AllocLimit` and `OutputLimit` are `LimitError` with Name "pixels", "scratch bytes" and "output
bytes" (`ErrLimit`); `Cancelled` is `context.Canceled`; `OutOfMemory` is new. Unlike Go, a zero
limit is a real cap, not "use the default" (Rust has no negative limits, so Go's negative-limit
error does not exist). No call writes to `dst` once a limit or argument error is returned.
`PixelLimit`, `AllocLimit` and `OutputLimit` carry `got` and `max`.
