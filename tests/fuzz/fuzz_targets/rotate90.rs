#![no_main]
use libfuzzer_sys::fuzz_target;

include!("common.rs");

// data = [w][h][ch][clockwise][pixels...], the ranges of raster-go's FuzzRotate90. Every sample
// lands where the definition puts it, so the SIMD tile kernel (grey, AVX-512) is held to it on the
// full 16x16 tiles of any size from 16 up, and the scalar edge tiles around them.
fuzz_target!(|data: &[u8]| {
    check_tier();
    let Some((&[w, h, ch, cw], rest)) = data.split_first_chunk::<4>() else {
        return;
    };
    let ch = 1 + 2 * usize::from(ch & 1);
    let (w, h, cw) = (
        usize::from(w) % 50 + 1,
        usize::from(h) % 50 + 1,
        cw & 1 == 1,
    );
    let (src, ss, out, ds) = call(rest, (w, h, ch, 1), (h, w, ch, h % 3), |d, s| {
        raster::rotate90(d, s, cw, &limits(), &no_cancel())
    });
    for y in 0..w {
        for x in 0..h {
            let (sx, sy) = if cw { (y, h - 1 - x) } else { (w - 1 - y, x) };
            for c in 0..ch {
                assert_eq!(
                    out[y * ds + x * ch + c],
                    src[sy * ss + sx * ch + c],
                    "({x},{y},{c})"
                );
            }
        }
    }
});
