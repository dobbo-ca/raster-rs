#![no_main]
use libfuzzer_sys::fuzz_target;

include!("common.rs");

// data = [w][h][dw][dh][ch][pixels...], the ranges of raster-go's FuzzResample. It must succeed
// and leave the padding alone; at the source's own size it is the identity, exactly. The SIMD tier
// (the vertical pass, AVX-512) equals the scalar code bit for bit: `force_scalar` (fuzzing builds
// only) runs the same call with every kernel scalar.
fuzz_target!(|data: &[u8]| {
    check_tier();
    let Some((&[w, h, dw, dh, ch], rest)) = data.split_first_chunk::<5>() else {
        return;
    };
    let ch = 1 + 2 * usize::from(ch & 1);
    let (w, h) = (usize::from(w) % 90 + 1, usize::from(h) % 90 + 1);
    let (dw, dh) = (usize::from(dw) % 120 + 1, usize::from(dh) % 120 + 1);
    let shape = ((w, h, ch, 2), (dw, dh, ch, dw % 3));
    let run = |scalar| {
        raster::force_scalar(scalar);
        let (_, _, out, _) = call(rest, shape.0, shape.1, |d, s| {
            raster::resample(d, s, &limits(), &no_cancel())
        });
        raster::force_scalar(false);
        out
    };
    assert!(run(false) == run(true), "tier differs from scalar");
    let (src, ss, out, ds) = call(rest, (w, h, ch, 2), (w, h, ch, 0), |d, s| {
        raster::resample(d, s, &limits(), &no_cancel())
    });
    for y in 0..h {
        assert_eq!(
            out[y * ds..][..w * ch],
            src[y * ss..][..w * ch],
            "identity row {y}"
        );
    }
});
