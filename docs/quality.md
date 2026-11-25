# Real-file quality check

Status: manual, not run in CI. `oracle -quality` (`tests/parity/go/oracle/quality.go`, the raster-go
snapshot in `tests/parity/go/upstream`) and `examples/quality.rs` run the same seven operations on the
pixels of real slide tiles. The outputs must be byte-identical.

## Running it

1. Take JPEG tiles from the OpenSlide test data (https://openslide.cs.cmu.edu/download/openslide-testdata/),
   for example Aperio `CMU-1-Small-Region.svs` and `CMU-1.svs`, `Leica-1.scn`, `Leica-Fluorescence-1.scn`,
   Generic-TIFF `CMU-1.tiff`, `Philips-4.tiff` and `Ventana-1.bif`. Decode each tile once to RGB and write
   `<stem>.ppm` (`P6\n<w> <h>\n255\n`, then the samples) into one directory, with any JPEG decoder (for
   example the `ppm` example of jpeg-rs). Both sides read these files, so a decoder difference cannot show
   up as a raster difference.
2. Build the oracle in `tests/parity/go/oracle` and run `oracle -quality <ppms> <go-out>`.
3. Run `cargo run --release --example quality -- <ppms> <rs-out>`, then
   `RASTER_SIMD=scalar cargo run --release --example quality -- --threads 1 <ppms> <rs-scalar-out>`.
   The first output line is `tier: <tier>`. With `RASTER_EXPECT_AVX512=1` (and no `RASTER_SIMD`), `quality`
   exits 1 if `raster::tier()` is not `avx512`.
4. `diff -r <go-out> <rs-out>` and `diff -r <go-out> <rs-scalar-out>` must print nothing.

## Operations

Per tile `<stem>.ppm` (w x h, RGB, rows tight: stride is `w * channels`), seven outputs:

| Output | Operation |
|---|---|
| `<stem>.s2rgb.ppm` | `scale2to1` of the RGB tile, (w/2, h/2). The jpeg-rs pyramid reference. Skipped if w or h is under 2. |
| `<stem>.luma.pgm` | `luma` of the RGB tile. |
| `<stem>.s2gray.pgm` | `scale2to1` of the luma plane. Skipped if w or h is under 2. |
| `<stem>.rsrgb.ppm` | `resample` of the RGB tile to (max(1, w*3/7), max(1, h*3/7)). |
| `<stem>.rsup.pgm` | `resample` of the luma plane up to (w*3/2, h*3/2). Skipped above 1 << 22 pixels. |
| `<stem>.rot.ppm` | `rotate90` clockwise of the RGB tile, (h, w). |
| `<stem>.rotg.pgm` | `rotate90` clockwise of the luma plane, (h, w): the grey path, which takes the AVX-512 16x16 tile kernel. |

Both sides write exactly `P6\n<w> <h>\n255\n` (RGB) or `P5\n<w> <h>\n255\n` (grey), then the samples. A raster
error, or a PPM that is not `P6` with maxval 255, a tile above 1 << 26 pixels, or a length that does not match
its size, stops the run with the file name and a non-zero exit. Limits: Go `MaxScratchBytes` 1 GiB; Rust
`max_alloc` 1 GiB, `max_threads` from `--threads`, the rest default. The Rust side runs every resample
through one `raster::Resampler` kept across all tiles (so reuse of the kept buffers across different
shapes is compared against Go's fresh call); the Go side calls the free function. Both sides take regular
files only from the PPM directory (a symlink is skipped).
