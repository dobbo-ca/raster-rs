# Differences from Go

- `dst` is a `PlaneMut`, so overlap with `src` is a compile error rather than a documented rule.
- `resample` refuses a dimension above `i32::MAX` with `Invalid`; Go's int32 tap tables wrap there.
- `Impl(kernel)` is `tier()`, one value for the host rather than per kernel name. It can return
  `"avx512"` (x86_64 with AVX2, AVX-512 F, BW and VBMI), `"avx2"`, `"neon"` or `"generic"` (scalar code), as
  the host has; `RASTER_SIMD=avx2` caps the tier at AVX2 and `RASTER_SIMD=scalar` forces scalar code.
- `Resampler` is Rust-only: Go has no reusable resample state. Its output (bytes and errors) equals Go's
  `Resample`, for any history of shapes.
- Go has no goroutines in this package; the threading, `max_alloc` ledger and `max_threads` are
  new. Go's `MaxScratchBytes` is `max_alloc` here.
