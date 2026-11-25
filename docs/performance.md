# Performance

How to build and call this crate for speed. The measured numbers are in [benchmarks.md](benchmarks.md).

## Build profile

A dependent crate's own release profile applies to this code, so the profile that counts is the
consumer's. For a program that scales tiles:

```toml
[profile.release]
lto = "fat"
codegen-units = 1
panic = "abort"
```

```sh
RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls" cargo build --release
```

- `lto = "fat"` and `codegen-units = 1` let the kernels and the loops that call them inline across
  modules and across the crate boundary. A dependent that only wants the one-unit build of this
  crate can set `[profile.release.package.raster] codegen-units = 1`.
- `panic = "abort"` removes the unwinding tables. It also changes one thing here: a panic on a pool
  thread is re-raised on the calling thread (rayon), and with `abort` the process stops at the
  panic. The crate does not panic on any input (clippy denies `unwrap`, `expect` and `panic` in its
  non-test code), so this only matters for a bug.
- `-C target-cpu=x86-64-v4` lets the compiler use AVX-512 (F, BW, CD, DQ, VL) everywhere, not only
  in kernels. **Use it only when every machine that runs the binary has AVX-512.** On a CPU without
  it the program dies with `SIGILL` (illegal instruction) at the first such instruction, with no
  message. The benchmark host (AMD Ryzen 7 9800X3D) has AVX-512; GitHub-hosted runners may not. The
  default build needs no flag and runs everywhere.
- `-C target-feature=+false-deps-tzcnt,+false-deps-bls` goes with it. On Zen 4 and Zen 5 `tzcnt`,
  `blsr`, `blsi` and `blsmsk` wait for the last write to their output register, and LLVM breaks that
  wait only under `target-cpu=znver4`/`znver5` tuning, not under `x86-64-v4`. The jpeg-rs encoder
  ran 30 to 60% slower without the two features. This crate has no `tzcnt` loop that we know of, so
  the flags are most likely a no-op here; they are in the recommendation so every crate of a
  program is built the same way, and `bench.yml`'s v4 column uses them. rustc warns that it does not
  know the two features and passes them to LLVM, which does.
- aarch64: do not set any of these flags (they are x86 only). NEON is part of the aarch64 baseline,
  so the default target already uses it; `lto`, `codegen-units` and `panic` apply as above. Build
  aarch64 binaries separately and pick their `target-cpu` for the aarch64 machines.

## SIMD dispatch at run time

The SIMD tier is chosen at run time, not by the build flags: AVX-512 when the CPU has it (F, BW,
VBMI), else AVX2 (or NEON on aarch64), whatever `target-cpu` says (`raster::tier()`,
[safety.md](safety.md)). `RASTER_SIMD=avx2` caps x86_64 at AVX2 (the bench's AVX2 column, the
tests); `RASTER_SIMD=scalar` forces the scalar tier, which is the reference the SIMD kernels are tested
against; the output is the same in every tier. What a v4 build changes is the code the compiler
generates for the scalar paths (the kernels a tier lacks, the horizontal resample pass, RGB
`rotate90` and the tails of the SIMD rows). The v4 column of
[benchmarks.md](benchmarks.md) shows what that gives against Go's `GOAMD64=v4` build, and the `ci`
stage runs `cargo test --release --lib --test raster` with these flags, so the v4 build is held to
the same golden bytes.

## Reusing a `Resampler`

`scale2to1`, `luma` and `rotate90` write only into the caller's planes and allocate nothing at
`max_threads` 1. `resample` allocates its tap tables, a weight row, an intermediate plane and the
accumulator rows on every call. A worker that resamples tile after tile should keep one `Resampler`
per thread and call `Resampler::resample` instead; the free `resample` is the same call on a fresh
`Resampler`.

```rust
use raster::{Limits, Resampler};
let limits = Limits { max_threads: 1, max_alloc: 256 << 20, ..Limits::default() };
let mut rs = Resampler::default();
// for each tile: rs.resample(&mut dst, &src, &limits, &cancel)?;
```

- It keeps the tap tables of the last `(source, destination)` size of each axis, and rebuilds an
  axis only when that pair changes (tiles of one size, scaled alike, never rebuild). It also keeps
  the weight row, the intermediate plane and the accumulator rows.
- The output of every call is byte-identical to the free `resample` for every input, thread count
  and sequence of earlier calls. `tests/reuse_alloc.rs` runs one `Resampler` over shrinking, growing
  and repeated shapes, at 1 and 4 threads, and compares; the `ops` fuzz target runs one reused
  `Resampler` over the input and a derived destination size, at 1 and 4 threads, and compares each call with the free function.
  A unit test in `src/lib.rs` (`reused_resampler_peak_within_max_alloc`) checks the ledger's peak against `max_alloc` on a reused one.
- With `max_threads` 1 the second call of the same shape makes zero allocator calls (the same test
  counts them for every op, `Resampler::resample` included), provided its `max_alloc` is no tighter
  and its `max_threads` no lower than in the call that built the buffers; otherwise the call trims
  the buffers the ledger cannot hold and allocates them again. A larger shape grows the buffers once;
  a smaller one reuses a part of them.
- The bench example times the free `resample`, a fresh `Resampler` on every call, so its tables
  measure the fresh-call path only, not buffer reuse.
- With more threads a call also makes a few small allocations that are not the `Resampler`'s: the
  boxed job rayon makes for each extra worker, and the list of slots and band results in
  `par_rows_with`. They are bounded by the thread count and do not grow with the image. The
  `Resampler` keeps its accumulator rows: at least one per band a call ran, plus any an earlier call left.
- `max_alloc` is still the peak of each call. The buffers a `Resampler` keeps are charged against
  it at the start of every call, at their capacity (not the length), together with what the call
  must grow, and before anything grows. Go's up-front check of the single-thread need is unchanged
  and applies on every call. The kept buffers can be bigger than a smaller call needs (a big tile
  came first), so if charging them would pass `max_alloc`, the call frees the buffers that are bigger
  than it needs and charges again; the second charge is at most the fresh need. A call that passes
  as the free function therefore passes on any `Resampler`, and one that fails fails with the same
  `Error::AllocLimit`. After an `AllocLimit` the `Resampler` holds nothing; after any other error it
  is still usable.
- Memory stays with the `Resampler` until it drops: the buffers of the largest shape it has seen,
  unless a call under a tighter `max_alloc` freed them. For one 2550x3300 grey page scaled to half
  that is the intermediate plane (about 4 MB) plus the taps and the rows, so a worker's `Resampler`
  costs about as much as one call's peak.

## PGO and BOLT

Profile-guided optimisation and BOLT apply to the final binary, so the consumer does them (the
program that links this crate); the library does not change. `pgo.yml` (dispatch only, no secrets)
measures how much they gain on the bench example: it builds the example with `cargo-pgo`, trains it
on its own 2550x3300 page (`bench 40 1` and `bench 40 0`, so the single-threaded and the threaded
paths are in the PGO profile; the BOLT profile is trained on `bench 40 1` only, so the "PGO + BOLT"
build is tuned for the single-threaded path), then times the default, the PGO and the PGO plus BOLT binaries with
`tests/parity/bench.sh` and prints the tables in the job summary. BOLT is optional there: if the
package is not available the job reports PGO only.

Rust ms per call (lower is better), a `pgo.yml` run before the AVX-512 tier, on a Ryzen 7
9800X3D; the change is against the default build:

| kernel | default (1 thread) | PGO | PGO + BOLT | default (default threads) | PGO | PGO + BOLT |
|---|---|---|---|---|---|---|
| luma | 0.452 | 0.447 (-1%) | 0.447 (-1%) | 0.096 | 0.099 (+3%) | 0.095 (-1%) |
| resample | 21.615 | 35.578 (+65%) | 32.595 (+51%) | 4.324 | 6.087 (+41%) | 5.916 (+37%) |
| rotate | 11.144 | 10.315 (-7%) | 10.098 (-9%) | 2.414 | 0.956 (-60%) | 0.972 (-60%) |
| scale-gray | 0.114 | 0.112 (-2%) | 0.114 (0%) | 0.035 | 0.037 (+6%) | 0.039 (+11%) |
| scale-rgb | 2.023 | 2.834 (+40%) | 2.013 (-1%) | 0.406 | 0.434 (+7%) | 0.400 (-2%) |

Result: PGO is not a gain for this crate as a whole. The SIMD kernels (luma, scale-gray) are the same
within noise. PGO makes resample much slower (the cause is not analysed),
and BOLT gets back only part of it. Only rotate gains, mostly with threads. Do not
use a PGO profile trained on this bench for a program; a program that wants PGO must train on its own
tile stream and check resample against the default build.

