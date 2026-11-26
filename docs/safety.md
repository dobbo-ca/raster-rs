# Safety and SIMD

Go has assembly for two kernels, and so does this crate on AVX2 and NEON: the grey `scale2to1` and
`luma`. The AVX-512 tier (x86_64) adds the RGB `scale2to1`, the grey `rotate90` tile and the
vertical pass of `resample`. Everything else (RGB `rotate90`, the horizontal `resample` pass, the
tails of every row) is scalar. The scalar reference is always compiled and is the oracle: every tier
is bit-exact with it for every input. The tier is detected once (`is_x86_feature_detected!`,
`is_aarch64_feature_detected!`), best first: AVX-512, then AVX2, on x86_64. `RASTER_SIMD=scalar` forces
scalar (Go's `RASTERGO_CPU=generic`); `RASTER_SIMD=avx2` caps x86_64 at AVX2 (for the bench's AVX2
column and for tests; on other targets it is ignored); `RASTER_SIMD=avx512` panics on a CPU without
AVX-512 F, BW and VBMI (for CI under an emulator). `raster::tier()` (a query, so it takes no
`Limits`) reports `"avx512"`, `"avx2"`, `"neon"` or `"generic"` (Go's `Impl`). A tier lacking a kernel
has `None` for it in `Kernels` and the scalar code runs it. The kernels are safe `#[target_feature]`
fns; `unsafe` is only in the per-tier dispatch (calling a kernel after detection) and the array-ref
load/store helpers. The kernels do whole lanes and the scalar row finishes the tail. Unit tests
compare every kernel of every tier the host has with the scalar code, bit for bit, on random and
all-0xff/0 data, edge widths and unaligned starts. They assert AVX2 (x86_64) or NEON (aarch64) was
detected, and AVX-512 when `RASTER_EXPECT_AVX512=1` or `RASTER_SIMD=avx512` (set either only on a
host with AVX-512; CI sets the first for the x86-64-v4 tests, which run only on such a host, and the
second in `avx512-sde`, under Intel SDE).

## AVX-512 tier (`src/simd/avx512.rs`)

Detected with `avx2`, `avx512f`, `avx512bw` and `avx512vbmi` (Zen 4/5, Ice Lake and later). A host with
only F and BW (Skylake-X) runs the AVX2 tier. No masked loads or stores, and no AVX-512VL.

| Kernel | Step | How |
|---|---|---|
| grey `scale2to1` (`box_gray`) | 32 outputs | the AVX2 kernel (the 64-wide AVX-512 one was dropped, see below) |
| `luma` | 32 pixels (96 bytes, read exactly) | two VPERMT2B gather the (R,G) and (B,x) words, VPMADDUBSW, VPMOVWB |
| RGB `scale2to1` (`box_rgb`) | 30 samples (10 pixels) | VPERMB lines up (a,b) byte pairs per row, VPMADDUBSW, add, VPMOVWB; the 32-byte store spills 2 bytes that the next step or the scalar tail overwrites |
| `rotate90` tile (`tile`, grey) | one 16x16 tile | 16 row loads into 4 zmm, two stages of VPERMT2B transpose, 16 row stores; edge tiles and RGB stay scalar |
| `resample` vertical (`vert`) | 64 columns | two taps per VPMADDWD on interleaved, widened rows, i32 accumulators, VPACKSSDW/VPACKUSWB clamp |

Not done: the horizontal `resample` pass. A tap row holds only `cnt` valid weights (the rest of the
row is stale or the next output's), so a per-output dot product needs a copy or a masked load, and a
vector across outputs needs gathers, which the safe-helper pattern does not allow. RGB `rotate90`
stays scalar: a 16x16 transpose of 3-byte elements has no clean VPERMB form. No AVX2 version of the
new kernels was written; on AVX2 they are scalar.

### Unsafe inventory

- Load/store helpers (each takes `&[u8; N]` or `&mut [u8; N]`, so the array type proves the bytes
  are in bounds; the access is unaligned): `ld128`, `st128` (SSE2, baseline), `ld256`, `st256`
  (AVX2), `ld512`, `st512` (AVX-512F). The AVX2 and NEON modules have their own.
- Dispatch fns, one per kernel (`luma`, `box_rgb`, `tile`, `vert`; the grey box is AVX2's `box_gray`,
  a safe `pub(super)` fn, so that only those two `detect` fns call it, after checking AVX2, is by convention inside `simd`, not enforced by the compiler; its SAFETY comment says so): each calls its
  `#[target_feature]` kernel under a SAFETY comment naming `detect`, which only hands out the fn
  pointer once all four features are present. `luma` and `box_rgb` trim the slices to the
  lanes they do. `tile` and `vert` pass the slices through. `tile` gets its 16-byte loads and stores through
  the checked `at`/`at_mut` helpers and returns false when one is out of the slice. `vert` gets its
  64-byte loads of the intermediate plane through `at` and returns a short count when one is out; its
  stores take no `at_mut` helper: `vert_kernel` walks the output with `out.as_chunks_mut::<64>()`, which
  yields only whole `&mut [u8; 64]` chunks, and each result goes through `st512` into one of them, so the array type
  is the proof; the 4 `__m512i` accumulators are register arrays, not memory.
- No other `unsafe`: the kernels and the index tables (`const fn`) are safe code.

### Kernels tried and dropped

From the CI benchmark (`bench.yml`) on a Ryzen 7 9800X3D: ms per call on a
2550x3300 page at 1 thread, the AVX-512 tier against the same binary with `RASTER_SIMD=avx2` (the
AVX2 tier, where `box_rgb`, `tile` and `vert` are scalar). A kernel that did not beat it was removed.

| Kernel | AVX-512 ms | AVX2 tier ms | AVX-512 / AVX2 | Decision |
|---|---|---|---|---|
| `box_gray` (64), scale-gray | 0.117 | 0.116 | 1.01 | dropped: no gain; the tier uses the AVX2 kernel |
| `luma` (32) | 0.284 | 0.458 | 0.62 | kept |
| `box_rgb` (30), scale-rgb | 0.275 | 2.025 (scalar) | 0.14 | kept |
| `rotate90` tile (grey), rotate | 0.551 | 14.803 (scalar) | 0.04 | kept |
| `resample` vertical, resample grey | 19.687 | 21.863 | 0.90 | kept |
| `resample` vertical, resample RGB | 48.376 | 54.978 | 0.88 | kept |
| `resample` horizontal | - | - | - | not written (see above) |
| `rotate90` RGB tile (rotate-rgb 15.505 vs 16.586, noise: both scalar) | - | - | - | not written (see above) |
