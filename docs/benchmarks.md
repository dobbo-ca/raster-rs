# Benchmark

`bench.yml` runs Go and Rust on a 2550x3300 page and prints a table of ms per call (see
`tests/parity/bench.sh`): Go in one configuration, since it has no goroutines and so is
single-threaded at any `GOMAXPROCS`, and Rust in two, at the default threads (`Limits::default()`)
and at 1 thread, each with its ratio to Go. The kernels are `scale-gray`, `scale-rgb`, `luma`,
`resample`, `resample-rgb`, `rotate` and `rotate-rgb`, the same rows from `examples/bench.rs` and
`oracle -bench`. The first table has two more columns: "Rust AVX2 ms (1 thread)", the same bench run
with `RASTER_SIMD=avx2` (the AVX-512 tier capped at AVX2) at 1 thread, and "AVX-512 / AVX2", the
1-thread default time over it. Below 1 the AVX-512 tier wins; that ratio decides which AVX-512
kernels stay ([safety.md](safety.md)). On a host without AVX-512 both columns are the AVX2 tier. The
runs (Go, Rust, Rust 1 thread, Rust AVX2 at 1 thread) go in turn for 5 rounds and each column keeps its
fastest round, because the shared runners vary by 20-30% from run to run. The table is in the job
summary.

A second table, "v4 builds", follows it: Go built with `GOAMD64=v4` (`oracle-v4`) and Rust built with
`RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls"` (`bench-v4`,
its own cargo cache mount, `raster-target-v4`). It has no AVX2 column. Its columns are Go v4 ms, Rust v4 at the default
threads, the ratio, Rust v4 at 1 thread and the ratio, with the same rule (the three-way interleave,
each column keeps its fastest round). The script checks that the v4 and default builds print the same output sizes (a check on the row set only: the size is
fixed per kernel, so it does not detect a miscompile); the `ci` stage runs the unit tests and `tests/raster.rs`
with the v4 flags. Why these flags and when they are safe: [performance.md](performance.md).
`pgo.yml` (dispatch only) times the default, PGO and PGO plus BOLT builds of the bench example; its
table is in [performance.md](performance.md). The bench job's timeout is 90 minutes:
the image builds Rust and Go twice and the timing loop runs both.

## Numbers

`bench.yml` runs on a GitHub-hosted `ubuntu-24.04` runner, which has fewer cores than the host of the
numbers below and may lack AVX-512. On such a runner `bench.sh` times the default build in the v4
columns and says so on stderr. Do not read a difference from the numbers below as a regression.

Hosted runner baseline (2026-11-25, bench run 45705985357, `ubuntu-24.04`, 2 vCPUs, AVX-512
present, so the v4 columns are real v4 builds), 2550x3300 page, ms per call, fastest of 5 rounds:

| kernel | Go ms | Rust ms (default threads) | Rust / Go | Rust ms (1 thread) | Rust / Go | Rust AVX2 ms (1 thread) | AVX-512 / AVX2 |
|---|---|---|---|---|---|---|---|
| luma | 1.151 | 0.573 | 0.50 | 0.856 | 0.74 | 1.002 | 0.85 |
| resample | 48.179 | 32.278 | 0.67 | 34.872 | 0.72 | 37.727 | 0.92 |
| resample-rgb | 133.102 | 77.424 | 0.58 | 83.108 | 0.62 | 92.554 | 0.90 |
| rotate | 40.057 | 1.295 | 0.03 | 1.453 | 0.04 | 28.131 | 0.05 |
| rotate-rgb | 50.215 | 36.507 | 0.73 | 37.727 | 0.75 | 38.934 | 0.97 |
| scale-gray | 0.245 | 0.168 | 0.69 | 0.192 | 0.79 | 0.191 | 1.01 |
| scale-rgb | 9.292 | 0.715 | 0.08 | 1.110 | 0.12 | 4.232 | 0.26 |

The v4 build (default threads) against the default build: resample 31.6 against 32.3 ms,
resample-rgb 76.4 against 77.4 ms, the others within the noise. That gain is smaller than the
run-to-run noise, so v4 flags are not recommended on the strength of this run.

A `bench.yml` run of the current kernels (AVX-512 tier) on a Ryzen 7 9800X3D, 2550x3300 page, ms per call, fastest of 5 rounds:

| kernel | Go ms | Rust ms (default threads) | Rust / Go | Rust ms (1 thread) | Rust / Go | Rust AVX2 ms (1 thread) | AVX-512 / AVX2 |
|---|---|---|---|---|---|---|---|
| luma | 0.535 | 0.065 | 0.12 | 0.273 | 0.51 | 0.455 | 0.60 |
| resample | 31.049 | 4.437 | 0.14 | 19.796 | 0.64 | 21.689 | 0.91 |
| resample-rgb | 80.046 | 8.860 | 0.11 | 46.922 | 0.59 | 53.014 | 0.89 |
| rotate | 17.507 | 0.168 | 0.01 | 0.542 | 0.03 | 14.517 | 0.04 |
| rotate-rgb | 18.463 | 3.883 | 0.21 | 16.551 | 0.90 | 16.244 | 1.02 |
| scale-gray | 0.138 | 0.042 | 0.30 | 0.112 | 0.81 | 0.113 | 0.99 |
| scale-rgb | 5.228 | 0.067 | 0.01 | 0.275 | 0.05 | 2.014 | 0.14 |

v4 builds, same run (Go `GOAMD64=v4`, Rust `-C target-cpu=x86-64-v4` with the false-deps features):

| kernel | Go v4 ms | Rust v4 ms (default threads) | Rust v4 / Go v4 | Rust v4 ms (1 thread) | Rust v4 / Go v4 |
|---|---|---|---|---|---|
| luma | 0.543 | 0.060 | 0.11 | 0.252 | 0.46 |
| resample | 33.649 | 4.227 | 0.13 | 20.256 | 0.60 |
| resample-rgb | 88.473 | 8.820 | 0.10 | 47.645 | 0.54 |
| rotate | 17.381 | 0.173 | 0.01 | 0.554 | 0.03 |
| rotate-rgb | 17.893 | 3.807 | 0.21 | 16.253 | 0.91 |
| scale-gray | 0.138 | 0.035 | 0.26 | 0.094 | 0.68 |
| scale-rgb | 5.218 | 0.063 | 0.01 | 0.274 | 0.05 |

An earlier run (before the 64-wide grey box was dropped),
gave the same picture (for example resample 19.687 ms and rotate 0.551 ms at 1 thread); the tables
above are the later run. The runner's numbers vary by 20-30% from run to run.

Before the AVX-512 tier, default threads / 1 thread, Rust ms:
luma 0.100 / 0.452, resample 4.988 / 21.751, rotate 2.832 / 11.172, scale-gray 0.043 / 0.115,
scale-rgb 0.476 / 2.013 (the `resample-rgb` and `rotate-rgb` rows were added later).
