# Fuzzing

`tests/fuzz/` (cargo-fuzz, run with `--fuzz-dir tests/fuzz`) has `scale2to1`, `luma`, `rotate90` and `resample`, the ports of Go's four
`Fuzz*` targets, and `ops`, which feeds the parity input format (hostile sizes, strides, channel
counts, limits) to every entry point and checks that 1 and 4 threads give the same result and
bytes (the crate is built with `cfg(fuzzing)`, which lowers the band size so small inputs split).
The first three compare the output with the definition, so the SIMD tier this CPU runs is checked
against the formula for any width, any row padding and any tail (`scale2to1` covers the AVX-512
RGB box, `rotate90` its 16x16 tile); `scale2to1`, `luma`, `rotate90` and `resample` fail if no
SIMD tier was detected. `resample` checks the identity at the source's own size and, in a fuzzing
build, runs the same call again with `raster::force_scalar(true)` (a hook that exists only under
`--cfg fuzzing`) and requires the same bytes, which holds the AVX-512 vertical pass to the scalar
code. `fuzz.yml` runs every target with the best tier, and `luma` also with
`RASTER_SIMD=avx2`: it is the only target with an avx2 leg. The others have none because AVX2 has no
tile or vert kernel (`rotate90`, `resample`: that leg would be scalar only), and the grey box is the same kernel in both
tiers and the RGB box is scalar on AVX2 (`scale2to1`). With `RASTER_EXPECT_AVX512=1` (set it by hand on an AVX-512 host), `check_tier` (every target except `ops`, which does not call it) fails if the best-tier leg is not AVX-512. The runs use `-rss_limit_mb=2048 -malloc_limit_mb=512`, and the two kernel targets run again on a native arm64
runner (`fuzz-arm64`, the same `fuzz` stage and `cargo +nightly fuzz run` command, with the sanitizer). Seeds are Go's `testdata/fuzz/*/seed1` (with
the pixels of `fixture.Page(1, ...)`) in `tests/fuzz/seeds/<target>`, and, for the AVX-512 and AVX2
kernels, one hand-written seed each (the header bytes, then 64 pixel bytes `(37 i + 11) mod 256` that the
target cycles through):

- `resample/seed-vert-grey`: `59 59 3f 09 00` = w 90, h 90, dw 64, dh 10, grey (each is the byte mod 90 / 120, plus 1;
  ch is 1 + 2 * (byte & 1)). A grey row of 64 destination columns is one full 64-column step of the
  vertical pass, and 90 to 10 rows gives many taps.
- `rotate90/seed-tile-grey`: `27 23 00 01` = w 40, h 36 (byte mod 50, plus 1), grey, clockwise: 2x2 full 16x16 tiles
  of the AVX-512 tile kernel, with scalar edge tiles around them.
- `scale2to1/seed-box-grey`: `80 04 00 01` = w 130 (byte mod 130, plus 2), h 6 (byte mod 6, plus 2), grey, pad 1:
  a 65-sample destination row, so two 32-byte steps of the grey box and a one-sample tail, on a source
  with 1 byte of row padding.
