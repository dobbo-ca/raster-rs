# Testing

Status: scalar, SIMD, threaded, with enforced resource limits, Go parity, fuzzing and a benchmark.
CI runs fmt, clippy (x86_64 and aarch64), the tests on the detected tier, with `RASTER_SIMD=avx2`
and with `RASTER_SIMD=scalar`, the x86-64-v4 release tests, an aarch64 release build (`test` job), the
tests on a native arm64 runner (`arm64` job: NEON, then scalar), and the Go oracle against
`examples/oracle.rs` on the same inputs, four times: default threads, scalar tier, one thread, AVX2
tier (`parity` job; see `docs/parity.md`). The real-file quality check (`docs/quality.md`, run by
hand) compares seven outputs per tile with Go. `fuzz.yml` runs the fuzz targets weekly, and `bench.yml` the Go and Rust
benchmark side by side.
SIMD tiers in CI: scalar, AVX2 (x86_64) and NEON (aarch64) natively; the AVX-512 kernels under Intel
SDE (`avx512-sde`: the unit tests and the kernel tests of `tests/raster.rs` on `-spr` with
`RASTER_SIMD=avx512`, which fails on a CPU without AVX-512), and natively when the hosted runner has it.

| Function | What |
|---|---|
| `scale2to1` | exact 2:1 box, `(a+b+c+d+2)>>2` |
| `resample` | separable Catmull-Rom, Q14 taps, i32 accumulation |
| `luma` | RGB to grey, `(38R+75G+15B+64)>>7` |
| `rotate90` | quarter turn, clockwise or not |


`tests/raster.rs` ports `raster_test.go`: exactness against the formulas at 24 widths and with row
padding, extremes, rotation, resample properties, invalid planes, limits, scratch bounds,
cancellation, and the 30 golden SHA-256 hashes (`tests/data/golden/hashes.txt`, copied from Go,
which generated them on amd64 with `-tags purego`). The fixtures (`tests/common`) reproduce Go's
xorshift pages exactly. Go's guard-page overread check has no scalar-Rust equivalent (slices are
bounds-checked). `TestImpl` is `tier_is_detected`.

The threading and limit tests: the output of every kernel is the same for 1 and several
thread counts (and 0, which is 1), and `resample`'s also for several `max_alloc`; the unit tests
read the charge ledger's peak against `max_alloc` and the live thread count against `max_threads`;
`tests/reuse_alloc.rs` runs one `Resampler` over many shapes and compares with the free `resample`; `tests/alloc.rs` counts every allocation (worker threads included) with a global allocator and
checks the peak against the need `resample` reports, at one and several threads;
`tests/shared_pool.rs` (its own binary, so no other test builds the pool first) checks that the
shared pool is not built by `max_threads: 1`, a caller's pool or a tiny image, and is built by the
first call that has work for a second thread.
