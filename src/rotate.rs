//! The quarter-turn kernel, one 16x16 tile at a time for cache locality.

use crate::limits::{Ctx, simd_row_work};
use crate::{Error, cancelled, simd};

const TILE: usize = 16;

/// Rotates a `sw * sh` plane into `dst` (`sh` wide, `sw` high). The caller has validated every length.
pub(crate) fn quarter(
    dst: &mut [u8],
    dst_stride: usize,
    src: &[u8],
    src_stride: usize,
    (sw, sh, ch): (usize, usize, usize),
    clockwise: bool,
    ctx: Ctx,
) -> Result<(), Error> {
    // Bands of whole tile rows of `dst` (`dst` row y is source column y); the last row of `dst`
    // may end short of the stride.
    let need = (sw - 1) * dst_stride + sh * ch;
    // The SIMD tile kernel (grey only) does the full 16x16 tiles; the loops below do the rest.
    let tile = if ch == 1 {
        simd::kernels().and_then(|k| k.tile)
    } else {
        None
    };
    // A band has about MIN_BAND_WORK ns of real work: a unit is 1 ns per pixel on the scalar path;
    // the tile kernel is about 27x faster, so its rows weigh `simd_row_work` (about 1/16 ns per
    // pixel, the same weight as the other SIMD kernels, a little heavier than measured).
    let row_work = if tile.is_some() {
        simd_row_work(sh)
    } else {
        sh
    };
    ctx.bud.par_rows_with(
        &mut dst[..need],
        (dst_stride, sw, row_work),
        (TILE, usize::MAX),
        |band, ys, ye| {
            for y0 in (ys..ye).step_by(TILE) {
                cancelled(ctx.cancel)?;
                for x0 in (0..sh).step_by(TILE) {
                    if let Some(t) = tile
                        && y0 + TILE <= ye
                        && x0 + TILE <= sh
                    {
                        // The top-left of the 16x16 source block that lands in this tile.
                        let (sy, sx) = if clockwise {
                            (sh - TILE - x0, y0)
                        } else {
                            (x0, sw - TILE - y0)
                        };
                        let d = &mut band[(y0 - ys) * dst_stride + x0..];
                        if t(
                            d,
                            dst_stride,
                            &src[sy * src_stride + sx..],
                            src_stride,
                            clockwise,
                        ) {
                            continue;
                        }
                    }
                    for y in y0..(y0 + TILE).min(ye) {
                        for x in x0..(x0 + TILE).min(sh) {
                            let (sy, sx) = if clockwise {
                                (sh - 1 - x, y)
                            } else {
                                (x, sw - 1 - y)
                            };
                            band[(y - ys) * dst_stride + x * ch..][..ch]
                                .copy_from_slice(&src[sy * src_stride + sx * ch..][..ch]);
                        }
                    }
                }
            }
            Ok(())
        },
    )
}
