#![no_main]
use libfuzzer_sys::fuzz_target;

include!("common.rs");

// data = [w][h][ch][pad][pixels...], the ranges of raster-go's FuzzScale2to1. The SIMD tier (grey
// on every tier, RGB on AVX-512) equals the definition (a+b+c+d+2)>>2 bit for bit, from any width
// (so any tail) and any row padding (so any alignment), and leaves the padding alone.
fuzz_target!(|data: &[u8]| {
    check_tier();
    let Some((&[w, h, ch, pad], rest)) = data.split_first_chunk::<4>() else {
        return;
    };
    let ch = 1 + 2 * usize::from(ch & 1);
    let (w, h, pad) = (
        usize::from(w) % 130 + 2,
        usize::from(h) % 6 + 2,
        usize::from(pad),
    );
    let (src, ss, out, ds) = call(
        rest,
        (w, h, ch, pad % 5),
        (w / 2, h / 2, ch, pad % 3),
        |d, s| raster::scale2to1(d, s, &limits(), &no_cancel()),
    );
    let at = |x: usize, y: usize, c: usize| u32::from(src[y * ss + x * ch + c]);
    for y in 0..h / 2 {
        for x in 0..w / 2 {
            for c in 0..ch {
                let sum = at(2 * x, 2 * y, c)
                    + at(2 * x + 1, 2 * y, c)
                    + at(2 * x, 2 * y + 1, c)
                    + at(2 * x + 1, 2 * y + 1, c);
                assert_eq!(
                    u32::from(out[y * ds + x * ch + c]),
                    (sum + 2) >> 2,
                    "({x},{y},{c})"
                );
            }
        }
    }
});
