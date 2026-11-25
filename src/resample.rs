//! The separable Catmull-Rom scaler: Q14 i16 taps, i32 accumulation, an 8-bit intermediate plane.
//! Every float product is its own statement-level operation (Rust never fuses), so the taps are
//! bit-identical to Go's.

use crate::limits::Ctx;
use crate::{Error, cancelled, simd};
use std::sync::{Mutex, PoisonError};

const ONE: i32 = 1 << 14; // Q14 unity

/// The taps of one axis, kept for the `(src_n, dst_n)` they were built for.
#[derive(Debug, Default)]
struct Taps {
    key: Option<(usize, usize)>, // None: not built, or a build did not finish
    start: Vec<i32>,             // first source index per output
    cnt: Vec<i32>,               // tap count per output
    w: Vec<i16>,                 // out * stride taps; only the first `cnt` of a row are read
    stride: usize,
}

/// The buffers of [`resample`](crate::resample()), kept between calls so a call of a shape seen
/// before allocates nothing at `max_threads` 1. See [`Resampler::resample`].
///
/// It keeps the tap tables of the last `(src, dst)` size of each axis (rebuilt only when that
/// changes), the weight row, the intermediate plane and the accumulator rows. Keep one per worker
/// thread; it is not shared between calls running at once (`resample` takes `&mut self`).
#[derive(Debug, Default)]
pub struct Resampler {
    hx: Taps,
    vy: Taps,
    fw: Vec<f64>, // the weight row `build` uses per axis
    mid: Vec<u8>,
    accs: Vec<Vec<i32>>, // one i32 row per band of the vertical pass
}

/// What one call needs of each buffer, in elements.
struct Need {
    dw: usize,
    dh: usize,
    size_w: usize,
    size_h: usize,
    fw: usize, // 0 if both tap tables are kept
    mid: usize,
    mrow: usize,
}

fn fscale(src_n: usize, dst_n: usize) -> f64 {
    (src_n as f64 / dst_n as f64).max(1.0)
}

/// The widest tap row; a row never has more taps than source samples.
fn tap_stride(src_n: usize, dst_n: usize) -> u64 {
    ((4.0 * fscale(src_n, dst_n)).ceil() as u64)
        .saturating_add(2)
        .min(src_n as u64)
}

/// What `run` allocates: the intermediate plane, both tap tables, the f64 weight row `build`
/// uses per axis and one i32 accumulator row. It saturates instead of overflowing.
pub(crate) fn scratch_bytes(sw: usize, sh: usize, dw: usize, dh: usize, ch: usize) -> u64 {
    let (sw, sh, dw, dh, ch) = (sw as u64, sh as u64, dw as u64, dh as u64, ch as u64);
    let ts_w = tap_stride(sw as usize, dw as usize);
    let ts_h = tap_stride(sh as usize, dh as usize);
    sh.saturating_mul(dw)
        .saturating_mul(ch)
        .saturating_add(dw.saturating_mul(ts_w.saturating_mul(2).saturating_add(8)))
        .saturating_add(dh.saturating_mul(ts_h.saturating_mul(2).saturating_add(8)))
        .saturating_add(ts_w.saturating_add(ts_h).saturating_mul(8))
        .saturating_add(dw.saturating_mul(ch).saturating_mul(4))
}

/// The Catmull-Rom kernel (a = -0.5).
fn cubic(x: f64) -> f64 {
    let x = x.abs();
    let x2 = x * x;
    let x3 = x2 * x;
    if x < 1.0 {
        (1.5 * x3) - (2.5 * x2) + 1.0
    } else if x < 2.0 {
        (-0.5 * x3) + (2.5 * x2) - (4.0 * x) + 2.0
    } else {
        0.0
    }
}

/// Grows `v` to at least `n` elements (all `x`) if it is shorter; a long enough `v` is left as it
/// is. The caller has charged the growth first.
fn ensure<T: Clone>(v: &mut Vec<T>, n: usize, x: T) -> Result<(), Error> {
    if v.len() < n {
        // Drop the old block first so a grow is free then alloc, never a realloc that holds both.
        *v = Vec::new();
        v.try_reserve_exact(n).map_err(|_| Error::OutOfMemory)?;
        v.resize(n, x);
    }
    Ok(())
}

/// The tap stride and the length of the tap table for one axis.
fn axis(src_n: usize, dst_n: usize) -> Result<(usize, usize), Error> {
    let st = tap_stride(src_n, dst_n) as usize;
    Ok((st, dst_n.checked_mul(st).ok_or(Error::OutOfMemory)?))
}

fn nearest(c: f64, src_n: usize) -> i64 {
    ((c + 0.5).floor() as i64).clamp(0, src_n as i64 - 1)
}

impl Taps {
    /// Builds the tables for `src_n` to `dst_n` unless they are already those. The memory is
    /// charged by `Resampler::run`, not here.
    fn build(&mut self, fw: &mut Vec<f64>, src_n: usize, dst_n: usize) -> Result<(), Error> {
        if self.key == Some((src_n, dst_n)) {
            return Ok(());
        }
        self.key = None;
        let (st, size) = axis(src_n, dst_n)?;
        ensure(&mut self.start, dst_n, 0)?;
        ensure(&mut self.cnt, dst_n, 0)?;
        ensure(&mut self.w, size, 0)?;
        ensure(fw, st, 0.0)?;
        self.stride = st;
        let fw = &mut fw[..st];
        let s = src_n as f64 / dst_n as f64;
        let f = fscale(src_n, dst_n);
        let sup = 2.0 * f;
        for i in 0..dst_n {
            let c = (i as f64 + 0.5) * s - 0.5;
            let mut lo = ((c - sup).ceil() as i64).max(0);
            let mut hi = ((c + sup).floor() as i64).min(src_n as i64 - 1);
            if hi < lo {
                // sliver at an edge: nearest sample
                lo = nearest(c, src_n);
                hi = lo;
            }
            let mut n = (hi - lo + 1) as usize;
            let mut sum = 0.0;
            for (k, w) in fw[..n].iter_mut().enumerate() {
                *w = cubic(((lo + k as i64) as f64 - c) / f);
                sum += *w;
            }
            if sum == 0.0 {
                // all taps on a zero crossing: nearest sample
                fw[0] = 1.0;
                sum = 1.0;
                n = 1;
                lo = nearest(c, src_n);
            }
            let row = &mut self.w[i * st..][..n];
            let (mut tot, mut big, mut best) = (0i64, 0usize, 0i16);
            for (k, (r, w)) in row.iter_mut().zip(fw.iter()).enumerate() {
                let q = (w / sum * f64::from(ONE) + 0.5).floor() as i64;
                *r = q as i16;
                tot += q;
                if k == 0 || *r > best {
                    (big, best) = (k, *r);
                }
            }
            row[big] = row[big].wrapping_add((i64::from(ONE) - tot) as i16); // taps sum to exactly 1.0
            self.start[i] = lo as i32;
            self.cnt[i] = n as i32;
        }
        self.key = Some((src_n, dst_n));
        Ok(())
    }

    /// Whether a buffer is bigger than the `dst_n` outputs and `size` taps need.
    fn excess(&self, dst_n: usize, size: usize) -> bool {
        self.start.capacity() > dst_n || self.cnt.capacity() > dst_n || self.w.capacity() > size
    }
}

fn clamp8(acc: i32) -> u8 {
    (acc >> 14).clamp(0, 255) as u8
}

impl Resampler {
    /// What the call needs, given what is kept: the buffers it must grow and the tables it must
    /// rebuild.
    fn need(
        &self,
        (sw, sh, dw, dh): (usize, usize, usize, usize),
        ch: usize,
    ) -> Result<Need, Error> {
        let (st_w, size_w) = axis(sw, dw)?;
        let (st_h, size_h) = axis(sh, dh)?;
        let mrow = dw * ch;
        let fw = if self.hx.key == Some((sw, dw)) {
            0
        } else {
            st_w
        };
        let fw = fw.max(if self.vy.key == Some((sh, dh)) {
            0
        } else {
            st_h
        });
        Ok(Need {
            dw,
            dh,
            size_w,
            size_h,
            fw,
            mid: sh.checked_mul(mrow).ok_or(Error::OutOfMemory)?,
            mrow,
        })
    }

    /// The bytes this call holds: every kept buffer at its capacity, or at what the call needs if
    /// that is more (the growth is charged with it, before it happens).
    fn held_bytes(&self, n: &Need) -> u64 {
        let b = |cap: usize, want: usize, elem: usize| {
            (cap.max(want) as u64).saturating_mul(elem as u64)
        };
        let acc = if self.accs.is_empty() {
            (n.mrow as u64).saturating_mul(4)
        } else {
            self.accs.iter().fold(0u64, |s, a| {
                s.saturating_add((a.capacity() as u64).saturating_mul(4))
            })
        };
        [
            b(self.hx.start.capacity(), n.dw, 4),
            b(self.hx.cnt.capacity(), n.dw, 4),
            b(self.hx.w.capacity(), n.size_w, 2),
            b(self.vy.start.capacity(), n.dh, 4),
            b(self.vy.cnt.capacity(), n.dh, 4),
            b(self.vy.w.capacity(), n.size_h, 2),
            b(self.fw.capacity(), n.fw, 8),
            b(self.mid.capacity(), n.mid, 1),
            acc,
        ]
        .into_iter()
        .fold(0u64, u64::saturating_add)
    }

    /// Frees every kept buffer that is bigger than the call needs (and the accumulator rows).
    fn trim(&mut self, n: &Need) {
        if self.hx.excess(n.dw, n.size_w) {
            self.hx = Taps::default();
        }
        if self.vy.excess(n.dh, n.size_h) {
            self.vy = Taps::default();
        }
        if self.fw.capacity() > n.fw {
            self.fw = Vec::new();
        }
        if self.mid.capacity() > n.mid {
            self.mid = Vec::new();
        }
        self.accs.clear();
    }

    /// Resamples; the caller has validated planes and bounded `scratch_bytes`. The kept buffers
    /// and the growth this call needs are charged to the call's `Budget` first, so the ledger
    /// holds what a fresh `Resampler` would, never more. If the kept buffers are why that fails,
    /// the ones bigger than this call needs are freed and it is charged again, so a call that
    /// passes on a fresh `Resampler` passes on any. The horizontal pass is split over rows of the
    /// intermediate plane and the vertical pass over rows of `dst`, each into as many bands as the
    /// budget allows threads, and the vertical bands also need an accumulator row each.
    pub(crate) fn run(
        &mut self,
        (dst, dst_stride): (&mut [u8], usize),
        (dw, dh): (usize, usize),
        (src, src_stride): (&[u8], usize),
        (sw, sh, ch): (usize, usize, usize),
        ctx: Ctx,
    ) -> Result<(), Error> {
        let (bud, cancel) = (ctx.bud, ctx.cancel);
        cancelled(cancel)?;
        let mrow = dw * ch;
        self.accs.retain(|a| a.capacity() >= mrow);
        let mut need = self.need((sw, sh, dw, dh), ch)?;
        let _held = match bud.charge(self.held_bytes(&need)) {
            Ok(c) => c,
            Err(_) => {
                self.trim(&need);
                need = self.need((sw, sh, dw, dh), ch)?;
                match bud.charge(self.held_bytes(&need)) {
                    Ok(c) => c,
                    Err(e) => {
                        *self = Resampler::default();
                        return Err(e);
                    }
                }
            }
        };
        let Resampler {
            hx,
            vy,
            fw,
            mid,
            accs,
        } = self;
        hx.build(fw, sw, dw)?;
        cancelled(cancel)?;
        vy.build(fw, sh, dh)?;
        let (hx, vy) = (&*hx, &*vy);
        ensure(mid, need.mid, 0)?;
        bud.par_rows(&mut mid[..need.mid], (mrow, sh, mrow), |band, y0, y1| {
            for y in y0..y1 {
                cancelled(cancel)?;
                let input = &src[y * src_stride..][..sw * ch];
                let out = &mut band[(y - y0) * mrow..][..mrow];
                for x in 0..dw {
                    let ws = &hx.w[x * hx.stride..][..hx.cnt[x] as usize];
                    let base = hx.start[x] as usize * ch;
                    for c in 0..ch {
                        let mut acc = ONE / 2;
                        for (k, &w) in ws.iter().enumerate() {
                            acc += i32::from(w) * i32::from(input[base + k * ch + c]);
                        }
                        out[x * ch + c] = clamp8(acc);
                    }
                }
            }
            Ok(())
        })?;
        // One accumulator row per band: the first is charged with the kept buffers, more only
        // while they fit, so the band count shrinks instead of the call failing.
        if accs.is_empty() {
            accs.push(Vec::new());
        }
        for a in accs.iter_mut() {
            ensure(a, mrow, 0)?;
        }
        let acc_bytes = (mrow as u64).saturating_mul(4);
        let want = bud.bands(dh, mrow, 1);
        let mut extra = Vec::new(); // the charges of the rows added below
        while accs.len() < want {
            match bud.charge(acc_bytes) {
                Ok(c) => extra.push(c),
                Err(_) => break,
            }
            let mut a = Vec::new();
            ensure(&mut a, mrow, 0)?;
            accs.push(a);
        }
        let mid = &mid[..need.mid];
        let bands = accs.len();
        // Each band takes a row for the length of its work and puts it back.
        let rows = Mutex::new(std::mem::take(accs));
        let rows = &rows;
        let need_dst = (dh - 1) * dst_stride + mrow;
        // The SIMD tier's vertical pass does whole lanes of a row; the code below finishes it.
        let vk = simd::kernels().and_then(|k| k.vert);
        let r = bud.par_rows_with(
            &mut dst[..need_dst],
            (dst_stride, dh, mrow),
            (1, bands),
            |band, y0, y1| {
                let Some(mut acc) = rows.lock().unwrap_or_else(PoisonError::into_inner).pop()
                else {
                    return Err(Error::OutOfMemory);
                };
                let r: Result<(), Error> = (|| {
                    let acc = &mut acc[..mrow];
                    for y in y0..y1 {
                        cancelled(cancel)?;
                        let ws = &vy.w[y * vy.stride..][..vy.cnt[y] as usize];
                        let first = vy.start[y] as usize;
                        let out = &mut band[(y - y0) * dst_stride..][..mrow];
                        let done = vk.map_or(0, |f| f(out, mid, mrow, first, ws));
                        let acc = &mut acc[done..];
                        acc.fill(ONE / 2);
                        for (k, &w) in ws.iter().enumerate() {
                            let row = &mid[(first + k) * mrow..][..mrow];
                            let w32 = i32::from(w);
                            for (a, &v) in acc.iter_mut().zip(&row[done..]) {
                                *a += w32 * i32::from(v);
                            }
                        }
                        for (o, &a) in out[done..].iter_mut().zip(&*acc) {
                            *o = clamp8(a);
                        }
                    }
                    Ok(())
                })();
                rows.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push(acc);
                r
            },
        );
        *accs = std::mem::take(&mut *rows.lock().unwrap_or_else(PoisonError::into_inner));
        r
    }
}
