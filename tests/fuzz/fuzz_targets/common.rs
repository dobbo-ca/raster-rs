// Shared by the differential targets (`include!`d; not a target itself).

use raster::{Limits, Plane, PlaneMut};
use std::sync::atomic::AtomicBool;

/// A runner without the SIMD tier would fuzz only the scalar one: fail instead.
fn check_tier() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let scalar = std::env::var_os("RASTER_SIMD").is_some_and(|v| v == "scalar");
        assert!(scalar || raster::tier() != "generic", "no SIMD tier");
        // RASTER_EXPECT_AVX512=1 (on an AVX-512 host) stops the auto leg quietly fuzzing AVX2.
        let avx2 = std::env::var_os("RASTER_SIMD").is_some_and(|v| v == "avx2");
        if std::env::var_os("RASTER_EXPECT_AVX512").is_some_and(|v| v == "1") && !scalar && !avx2 {
            assert_eq!(raster::tier(), "avx512", "no AVX-512 tier");
        }
        eprintln!("tier: {}", if scalar { "scalar" } else { raster::tier() });
    });
}

/// A `w*h*ch` plane with `pad` spare bytes per row, filled by cycling through `rest`.
fn source(rest: &[u8], w: usize, h: usize, ch: usize, pad: usize) -> (Vec<u8>, usize) {
    let stride = w * ch + pad;
    let n = (h - 1) * stride + w * ch;
    let pix = (0..n).map(|i| rest.get(i % rest.len().max(1)).copied().unwrap_or(0));
    (pix.collect(), stride)
}

/// The destination: 0xEE everywhere, so a write into the padding shows.
fn blank(w: usize, h: usize, ch: usize, pad: usize) -> (Vec<u8>, usize) {
    let stride = w * ch + pad;
    (vec![0xEE; (h - 1) * stride + w * ch], stride)
}

/// Runs `f` on planes of the given shapes; returns the source, its stride, the output, and the
/// output stride. The padding of the output is checked untouched.
fn call(
    rest: &[u8],
    (w, h, ch, pad): (usize, usize, usize, usize),
    (dw, dh, dch, dpad): (usize, usize, usize, usize),
    f: impl FnOnce(&mut PlaneMut, &Plane) -> Result<(), raster::Error>,
) -> (Vec<u8>, usize, Vec<u8>, usize) {
    let ((src, sstride), (mut out, dstride)) =
        (source(rest, w, h, ch, pad), blank(dw, dh, dch, dpad));
    let s = Plane {
        pix: &src,
        w,
        h,
        stride: sstride,
        channels: ch,
    };
    let mut d = PlaneMut {
        pix: &mut out,
        w: dw,
        h: dh,
        stride: dstride,
        channels: dch,
    };
    f(&mut d, &s).unwrap();
    for y in 0..dh - 1 {
        let at = y * dstride + dw * dch;
        assert!(
            out[at..at + dpad].iter().all(|&b| b == 0xEE),
            "padding written"
        );
    }
    (src, sstride, out, dstride)
}

fn limits() -> Limits {
    Limits::default()
}

fn no_cancel() -> AtomicBool {
    AtomicBool::new(false)
}
