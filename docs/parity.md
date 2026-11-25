# Go parity

The `parity` job (`.github/ci/Dockerfile`, target `parity`) runs the Go library and this crate on the same
inputs and does `diff -rq` on the outputs. It fails if the trees differ and if there are fewer than
1,000 outputs, so the diff is never empty by accident. The Rust side runs four times against the
same Go outputs: pass 1 at the default threads (`Limits::default().max_threads`) and the detected
SIMD tier, pass 2 with `RASTER_SIMD=scalar`, pass 3 with `oracle --threads 1`
(`Limits::max_threads` 1: no thread, no pool), pass 4 with `RASTER_SIMD=avx2` (the AVX2 tier; pass 1 is
AVX-512 where the host has it). The output bytes must equal Go's in every pass.

- `tests/parity/go/upstream/`: raster-go at f2cae8f, non-test sources (`raster.go`, `doc.go`, `internal/*` with
  the assembly), `go.mod`, LICENSE, NOTICE. Go runs the amd64 assembly in CI; this crate runs AVX-512 (AVX2 in pass 4).
- `tests/parity/go/oracle/`: `oracle -gen <inputs>` writes the inputs, `oracle <inputs> <outputs>` runs them,
  `oracle -bench N` times the kernels. Its own `go.mod` has `replace ... => ../upstream`.
- `tests/parity/case.rs`: the input parser and the call, included by `examples/oracle.rs` (the Rust side) and
  the `ops` fuzz target.

The same Go and Rust code can also run on real slide tiles decoded once to PPM (`docs/quality.md`:
`oracle -quality` and `examples/quality.rs`, outputs compared byte for byte).

## Input format

`[op][lim][flags][sch][dch]` then `sw sh sst dw dh dst dlen` as u32 LE (33 bytes), then the source
bytes. `op`: 0 `scale2to1`, 1 `luma`, 2 `rotate90` clockwise, 3 `rotate90` counter-clockwise, 4
`resample`. `lim` selects the limits: 0 the defaults with `max_alloc` at 64 MiB, 1 `max_pixels` 4095,
2 `max_alloc` 1 MiB, 3 `max_output` 1023, 4 `max_pixels` 4096 and `max_output` 1024 (so the
exact cap passes). `flags` bit 0 cancels before the call; `flags >> 1` = N > 0 makes the Nth cancel check fail. `sw sh sst` and `sch` describe the
source plane (the rest of the file is its buffer), `dw dh dst dch` the destination, whose buffer
is `dlen` zero bytes. A short file, an unknown `op` or a `dlen` above 2 MiB is `bad-input` (the
oracles bound their own allocations).

Per input the oracle writes `<name>.out` (the whole destination buffer, padding included), or
`<name>.err` (the class below) and `<name>.partial.out` (the destination left by the failed call:
Go returns it, and so does this crate, so both must match, which also shows that no call writes
after a limit or argument error). A cancel before the call leaves zero rows done. A cancel part-way
through uses N: Go wraps its context so the Nth `Err` call reports `Canceled`, and this crate's
`cancel-hook` feature (enabled by the `parity` job and the fuzz crate) fails the Nth check, so the
rows already written are compared (`cancel-mid-*` inputs). A call armed with the hook runs on the
calling thread whatever `max_threads` is, so the rows written before a cancel are the serial ones. Go checks per dst row (scale, luma), per
16-row tile band (rotate) and per src row then per dst row (resample); the oracle skips the two
extra checks resample makes before building its taps (they write nothing).

## Error classes

| class | Go | Rust |
|---|---|---|
| `invalid` | `ErrInvalid` | `Error::Invalid` |
| `limit-pixels` | `LimitError` "pixels" | `Error::PixelLimit` |
| `limit-scratch` | `LimitError` "scratch bytes" | `Error::AllocLimit` |
| `limit-output` | `LimitError` "output bytes" | `Error::OutputLimit` |
| `cancelled` | `context.Canceled` | `Error::Cancelled` |
| `bad-input` | the oracle's own check | the same |

`Error::OutOfMemory` has no Go counterpart; it would print as `other:<message>` and show up as a
diff. Go's negative-limit error does not exist in Rust (limits are unsigned), and a zero limit
means "default" in Go but a real cap here: the oracle always passes explicit non-zero limits.

## Ported Go tests

Go has 12 `Test*` functions in `raster_test.go`, 1 `Example` (`example_test.go`) and 4 `Fuzz*` targets, with seeds in
`testdata/fuzz` (in the Go repo). `tests/raster.rs` ports the tests. `oracle -gen` ports the parts that build
inputs, so Go supplies the expected bytes:

| Go | Go cases | Generated |
|---|---|---|
| `TestScale2to1` | 24 widths x 2 channels x 2 paddings | 96 `scale-*` |
| `TestLuma`, `TestLumaExtremes` | 24 widths x 2 paddings, 2 values | 48 + 2 `luma-*` |
| `TestRotate90` | 6 sizes x 2 channels x 2 directions | 24 `rotate-*` |
| `TestResample` | identity, 3 flat values x 5 sizes, ramp, per channel | 2 + 30 + 1 `resample-*` |
| `TestInvalid` | 7 bad planes as src, dst, and resample dst, wrong size | 22 `invalid*` |
| `TestLimits`, `TestResampleScratchLimit`, `TestResampleScratchExact` | pixels, output, exact cap, short buffer, scratch, 2 thin shapes | 3 + 1 + 1 + 2 `limit-*`, `resample-thin-*` |
| `TestCancel` | 4 kernels | 4 `cancel-*`, plus 25 `cancel-mid-*` (no Go counterpart: Go cancels only before the call) |
| `ExampleScale2to1` | 4x2 grey to 2x1, `35 55` | the doctest on `scale2to1` |
| `TestGolden` | 30 hashes | 30 `golden-*` (the bytes themselves, not hashes) |
| `FuzzScale2to1`, `FuzzLuma`, `FuzzRotate90`, `FuzzResample` | one seed each | 200 random parameter sets of each, with Go's ranges: 800 `fuzz-*` |
| `TestImpl` | tier table | `tier_is_detected` in `src/simd/mod.rs`, not a parity input |

Not ported to parity: the guard-page overread check (Rust slices are bounds-checked) and the
3000x2000 scratch-exactness shape (`tests/alloc.rs` measures the allocations; `tests/raster.rs` has the scratch accounting).

Mutations of 7 small valid calls (every operation, both channel counts): the truncations at
every length up to the header + 2 and 7 along the pixels, every header bit flip (except bits
12 to 20 of `dlen`, which only enlarge the output), and each length field set to 0, 1, 2, 2^31-1,
2^32-1, and the value -1 and +1: about 2,400 inputs of `mut*`. The ops fuzz seeds are copied in as
`seed-*`. `oracle -gen` prints the total.

## Benchmark

`tests/parity/bench.sh` and `.github/workflows/bench.yml`: `oracle -bench N` and `examples/bench.rs` time each
kernel N times on a 2550x3300 page (the page of `bench_test.go`) in one container and print ns
per call; `tests/parity/bench.sh` joins them into a table of ms per call and the Rust/Go ratio. Go has no goroutines, so
it has one configuration; Rust is run at the default threads (`Limits::default()`) and at 1 thread
(`bench <n> 1`), each with its ratio to Go, and at 1 thread with `RASTER_SIMD=avx2` (the AVX2 column and the AVX-512 / AVX2 ratio). The table has 7 kernel rows (`scale-gray`, `scale-rgb`, `luma`, `resample`, `resample-rgb`, `rotate`, `rotate-rgb`; see [benchmarks.md](benchmarks.md)). The four run in turn for 5 rounds (`tests/parity/bench.sh N
ROUNDS`), and each column keeps its fastest round per kernel.
