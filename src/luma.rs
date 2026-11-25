//! RGB to grey with integer BT.601 weights: `Y = (38R + 75G + 15B + 64) >> 7`.

use crate::limits::{Ctx, simd_row_work};
use crate::{Error, cancelled, simd};

pub(crate) fn plane(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    (w, h): (usize, usize),
    ctx: Ctx,
) -> Result<(), Error> {
    let simd = simd::kernels();
    // Bands of whole dst rows; the last row of `dst` may end short of the stride.
    let need = (h - 1) * dst_stride + w;
    // The SIMD tier does about 16 pixels per ns, so it weighs 16 pixels as one unit of work.
    let row_work = if simd.is_some() { simd_row_work(w) } else { w };
    ctx.bud.par_rows(
        &mut dst[..need],
        (dst_stride, h, row_work),
        |band, y0, y1| {
            for y in y0..y1 {
                cancelled(ctx.cancel)?;
                let out = &mut band[(y - y0) * dst_stride..][..w];
                let px = &src[y * src_stride..][..3 * w];
                // The SIMD tier does whole lanes; the scalar row finishes the rest.
                let done = simd.map_or(0, |k| (k.luma)(out, px));
                row(&mut out[done..], &px[3 * done..]);
            }
            Ok(())
        },
    )
}

/// The scalar reference: `out.len()` pixels from `3 * out.len()` bytes.
pub(crate) fn row(out: &mut [u8], src: &[u8]) {
    for (o, px) in out.iter_mut().zip(src.as_chunks::<3>().0) {
        let sum = 38 * u32::from(px[0]) + 75 * u32::from(px[1]) + 15 * u32::from(px[2]) + 64;
        *o = (sum >> 7) as u8;
    }
}
