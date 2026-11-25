# raster-rs

8-bit grey and RGB raster kernels in Rust: exact 2:1 scale, separable Catmull-Rom resample, RGB
to luma and quarter-turn rotation. A port of
[dobbo-ca/raster-go](https://github.com/dobbo-ca/raster-go) at f2cae8f. Output is bit-identical to
Go. Pure Rust; the only dependency is `rayon`. MIT licensed.

```toml
raster = { git = "https://github.com/dobbo-ca/raster-rs" }
```

## Usage

```rust
use raster::{Limits, Plane, PlaneMut};
use std::sync::atomic::AtomicBool;

fn main() -> Result<(), raster::Error> {
    let (w, h) = (640, 480);
    let src_bytes = vec![128u8; w * h];
    let mut dst_bytes = vec![0u8; (w / 2) * (h / 2)];

    // A caller in a 2-CPU, 1 GiB pod: at most 2 threads and 768 MiB for the call.
    let limits = Limits { max_threads: 2, max_alloc: 768 << 20, ..Limits::default() };
    let cancel = AtomicBool::new(false);

    let src = Plane { pix: &src_bytes, w, h, stride: w, channels: 1 };
    let mut dst = PlaneMut { pix: &mut dst_bytes, w: w / 2, h: h / 2, stride: w / 2, channels: 1 };
    raster::scale2to1(&mut dst, &src, &limits, &cancel)?;
    Ok(())
}
```

## Resource limits

Every function takes a `Limits` and a cancel flag; there is no unlimited call.

| Field | Default |
|---|---|
| `max_pixels` | `1 << 26` |
| `max_alloc` | `1 << 28` (256 MiB) |
| `max_output` | `1 << 28` |
| `max_threads` | `available_parallelism()`, 1 if unknown |
| `pool` | `None`: the crate's shared pool |

`max_alloc` is the enforced peak working memory of one call. `max_threads: 1` spawns no thread. A
caller's `pool` wins over the lazy shared pool. Details: [docs/limits.md](docs/limits.md).

## Docs

- [Limits, errors, cancellation](docs/limits.md)
- [Threading](docs/threading.md)
- [Safety and SIMD](docs/safety.md)
- [Go parity](docs/parity.md) and [tests](docs/testing.md)
- [Real-file quality check](docs/quality.md)
- [Fuzzing](docs/fuzzing.md)
- [Benchmarks](docs/benchmarks.md)
- [Performance: build profile, reusing a `Resampler`, PGO](docs/performance.md)
- [Differences from Go](docs/go-differences.md)
- [Dependencies](docs/dependencies.md)
