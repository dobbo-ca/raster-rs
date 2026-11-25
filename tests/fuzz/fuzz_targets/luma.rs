#![no_main]
use libfuzzer_sys::fuzz_target;

include!("common.rs");

// data = [w][h][pad][pixels...], the ranges of raster-go's FuzzLuma. The SIMD tier equals
// (38R+75G+15B+64)>>7 bit for bit, from any width and any row padding, and leaves the padding alone.
fuzz_target!(|data: &[u8]| {
    check_tier();
    let Some((&[w, h, pad], rest)) = data.split_first_chunk::<3>() else {
        return;
    };
    let (w, h, pad) = (
        usize::from(w) % 130 + 1,
        usize::from(h) % 6 + 1,
        usize::from(pad),
    );
    let (src, ss, out, ds) = call(rest, (w, h, 3, pad % 5), (w, h, 1, pad % 3), |d, s| {
        raster::luma(d, s, &limits(), &no_cancel())
    });
    for y in 0..h {
        for x in 0..w {
            let p = |c: usize| u32::from(src[y * ss + 3 * x + c]);
            assert_eq!(
                u32::from(out[y * ds + x]),
                (38 * p(0) + 75 * p(1) + 15 * p(2) + 64) >> 7,
                "({x},{y})"
            );
        }
    }
});
