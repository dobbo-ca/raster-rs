//! The exact 2:1 box kernel.

use crate::limits::{Ctx, simd_row_work};
use crate::{Error, cancelled, simd};

/// Writes `dw * dh` samples, each `(a+b+c+d+2)>>2` of a 2x2 block. The caller has validated every length.
pub(crate) fn box2to1(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    (dw, dh, ch): (usize, usize, usize),
    ctx: Ctx,
) -> Result<(), Error> {
    let rb = dw * ch;
    // The tier's row kernel: grey on every tier, RGB on the tiers that have one.
    let simd = simd::kernels().and_then(|k| if ch == 1 { Some(k.box_gray) } else { k.box_rgb });
    // Bands of whole dst rows; the last row of `dst` may end short of the stride.
    let need = (dh - 1) * dst_stride + rb;
    // The grey AVX2 tier does about 16 samples per ns, so it weighs 16 as one unit of work. The
    // AVX-512 RGB kernel reuses that weight, not a measured one: the bench (scale-rgb, 1 thread, 3825x1650
    // output samples: 0.275 ms, against 2.025 ms on the AVX2 tier, where the RGB box is scalar) is
    // about 23 samples per ns, so the real figure is higher and the weight was left as it was.
    let row_work = if simd.is_some() {
        simd_row_work(rb)
    } else {
        rb
    };
    ctx.bud.par_rows(
        &mut dst[..need],
        (dst_stride, dh, row_work),
        |band, y0, y1| {
            for y in y0..y1 {
                cancelled(ctx.cancel)?;
                let out = &mut band[(y - y0) * dst_stride..][..rb];
                let r0 = &src[2 * y * src_stride..][..2 * rb];
                let r1 = &src[(2 * y + 1) * src_stride..][..2 * rb];
                // The SIMD tier does whole lanes; the scalar row finishes the rest.
                let done = simd.map_or(0, |k| k(out, r0, r1));
                row(&mut out[done..], &r0[2 * done..], &r1[2 * done..], ch);
            }
            Ok(())
        },
    )
}

/// The scalar reference: `out.len()` samples from `2 * out.len()` bytes of each row. `ch` is 1 or
/// 3 (validated by the caller); a constant channel count lets the compiler drop the bounds checks.
pub(crate) fn row(out: &mut [u8], r0: &[u8], r1: &[u8], ch: usize) {
    if ch == 1 {
        row_c::<1>(out, r0, r1);
    } else {
        row_c::<3>(out, r0, r1);
    }
}

fn row_c<const C: usize>(out: &mut [u8], r0: &[u8], r1: &[u8]) {
    let rows = r0.chunks_exact(2 * C).zip(r1.chunks_exact(2 * C));
    for (px, (a, b)) in out.as_chunks_mut::<C>().0.iter_mut().zip(rows) {
        for (c, o) in px.iter_mut().enumerate() {
            let sum = u32::from(a[c]) + u32::from(a[c + C]) + u32::from(b[c]) + u32::from(b[c + C]);
            *o = ((sum + 2) >> 2) as u8;
        }
    }
}
