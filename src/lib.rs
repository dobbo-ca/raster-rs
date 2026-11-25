//! 8-bit grey and RGB raster kernels: exact 2:1 scale, separable Catmull-Rom resample, RGB to
//! luma and quarter-turn rotation.
//!
//! Output is bit-identical to raster-go. Every entry point takes [`Limits`] and a cancel flag.
//! Kernels write only into caller-allocated [`PlaneMut`]s; only [`resample`] allocates, and a
//! [`Resampler`] keeps those buffers between calls (see "Reusing a `Resampler`" in
//! `docs/performance.md`).
//!
//! The kernels have SIMD tiers chosen at run time (AVX-512, AVX2, NEON; see [`tier`]), each
//! bit-exact with the scalar code. `RASTER_SIMD=scalar` forces scalar, `RASTER_SIMD=avx2` caps
//! x86_64 at AVX2.
//!
//! # Resource limits
//!
//! Every entry point takes a [`Limits`]; there is no unlimited call. Its fields, with the error
//! each returns:
//!
//! | Field | Unit | Default | Over it |
//! |---|---|---|---|
//! | `max_pixels` | pixels (larger of source and destination) | `1 << 26` | [`Error::PixelLimit`] |
//! | `max_alloc` | bytes | `1 << 28` | [`Error::AllocLimit`] |
//! | `max_output` | bytes written to the destination | `1 << 28` | [`Error::OutputLimit`] |
//! | `max_threads` | threads | `std::thread::available_parallelism()`, else 1 | none: the call uses fewer threads |
//! | `pool` | [`ThreadPool`] | `None`: the crate's shared pool | none |
//!
//! `max_alloc` is the enforced peak working memory of one call, all its threads together: the
//! tap tables, the intermediate plane and one accumulator row for each band running at once are
//! charged before they are allocated and released when they are freed (Go's up-front scratch
//! check also stays). A [`Resampler`] also charges the buffers it keeps, at their capacity, at the
//! start of each call, and frees the ones bigger than the call needs if they would take it over
//! `max_alloc`, so a call that passes as [`resample`] passes on any `Resampler`. `max_threads` is
//! the most OS threads one call uses, the calling thread included; with `max_threads: 1` (or 0)
//! no thread is spawned and no pool is touched. The other threads are the workers of a
//! [`ThreadPool`]: the one in `Limits::pool` if the caller made one (it always wins), else the
//! crate's shared pool, built on the first call with `max_threads` above 1 that has work for a
//! second thread, with `std::thread::available_parallelism()` threads and never more. The output
//! bytes are the same for every `max_threads` and every pool.
//!
//! `cancel: &AtomicBool` is checked per row (per 16-row tile band for [`rotate90`]) by every
//! thread of the call; once it is set the call returns [`Error::Cancelled`] and `dst` may be
//! partly written: only rows finished before the flag was seen, and which rows depends on
//! `max_threads`.
//!
//! ```
//! use raster::{Limits, Plane, PlaneMut};
//! use std::sync::atomic::AtomicBool;
//!
//! # fn main() -> Result<(), raster::Error> {
//! let (w, h) = (64, 48);
//! let src_bytes: Vec<u8> = (0..w * h).map(|i| (i * 7 % 251) as u8).collect();
//! let mut dst_bytes = vec![0u8; (w / 2) * (h / 2)];
//! let src = Plane { pix: &src_bytes, w, h, stride: w, channels: 1 };
//! let mut dst = PlaneMut { pix: &mut dst_bytes, w: w / 2, h: h / 2, stride: w / 2, channels: 1 };
//! // A caller in a 2-CPU, 1 GiB pod.
//! let limits = Limits { max_threads: 2, max_alloc: 768 << 20, ..Limits::default() };
//! let cancel = AtomicBool::new(false);
//! raster::scale2to1(&mut dst, &src, &limits, &cancel)?;
//! // The first output is (0 + 7 + 197 + 204 + 2) >> 2.
//! assert_eq!(dst_bytes[0], 102);
//! # Ok(())
//! # }
//! ```
#![deny(
    unsafe_code,
    unsafe_op_in_unsafe_fn,
    clippy::undocumented_unsafe_blocks
)]
#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::todo,
        clippy::unimplemented
    )
)]

mod error;
mod limits;
mod luma;
mod parallel;
mod resample;
mod rotate;
mod scale;
mod simd;

pub use error::Error;
pub use limits::Limits;
pub use parallel::{ThreadPool, shared_pool_started};
pub use resample::Resampler;
#[cfg(fuzzing)]
#[doc(hidden)]
pub use simd::force_scalar;
pub use simd::tier;

use limits::{Budget, Ctx};
use std::sync::atomic::{AtomicBool, Ordering};

// The README's code blocks run as doc-tests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;

// Test hook: the Nth `cancelled` call on this thread fails (0 = off). The `cancel-hook` feature
// exposes it to the parity oracle and the `ops` fuzz target.
#[cfg(any(test, feature = "cancel-hook"))]
thread_local! {
    static CANCEL_AT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Makes the `n`th cancel check on this thread fail (0 = off). Parity and fuzz only.
///
/// # Errors
///
/// None: a test hook; it takes no `Limits`.
#[cfg(feature = "cancel-hook")]
#[doc(hidden)]
pub fn cancel_at_check(n: usize) {
    CANCEL_AT.with(|c| c.set(n));
}

pub(crate) fn cancelled(cancel: &AtomicBool) -> Result<(), Error> {
    #[cfg(any(test, feature = "cancel-hook"))]
    if CANCEL_AT.with(|c| {
        let n = c.get();
        c.set(n.saturating_sub(1));
        n == 1
    }) {
        return Err(Error::Cancelled);
    }
    if cancel.load(Ordering::Relaxed) {
        Err(Error::Cancelled)
    } else {
        Ok(())
    }
}

/// An 8-bit interleaved raster: `channels` is 1 (grey) or 3 (RGB), and row `y` starts at
/// `pix[y * stride]`.
#[derive(Debug, Clone, Copy)]
pub struct Plane<'a> {
    pub pix: &'a [u8],
    pub w: usize,
    pub h: usize,
    pub stride: usize,
    pub channels: usize,
}

/// A [`Plane`] to write into. Taking it as `&mut` makes overlap with the source impossible.
#[derive(Debug)]
pub struct PlaneMut<'a> {
    pub pix: &'a mut [u8],
    pub w: usize,
    pub h: usize,
    pub stride: usize,
    pub channels: usize,
}

/// What `validate` and `check` need of a plane.
#[derive(Clone, Copy)]
struct Dims {
    w: usize,
    h: usize,
    stride: usize,
    channels: usize,
    len: usize,
}

impl Plane<'_> {
    fn dims(&self) -> Dims {
        Dims {
            w: self.w,
            h: self.h,
            stride: self.stride,
            channels: self.channels,
            len: self.pix.len(),
        }
    }
}

impl PlaneMut<'_> {
    fn dims(&self) -> Dims {
        Dims {
            w: self.w,
            h: self.h,
            stride: self.stride,
            channels: self.channels,
            len: self.pix.len(),
        }
    }
}

impl Dims {
    fn validate(self) -> Result<(), Error> {
        if self.w == 0 || self.h == 0 || (self.channels != 1 && self.channels != 3) {
            return Err(Error::Invalid("bad width, height or channels"));
        }
        let row = self
            .w
            .checked_mul(self.channels)
            .ok_or(Error::Invalid("width overflows"))?;
        if self.stride < row {
            return Err(Error::Invalid("stride below row bytes"));
        }
        let need = (self.h - 1)
            .checked_mul(self.stride)
            .and_then(|n| n.checked_add(row))
            .ok_or(Error::Invalid("size overflows"))?;
        if self.len < need {
            return Err(Error::Invalid("buffer too short"));
        }
        Ok(())
    }

    fn pixels(self) -> u64 {
        self.w as u64 * self.h as u64
    }
}

/// Validates both planes and the limits; `scratch` is the bytes the call will allocate.
fn check(dst: Dims, src: Dims, scratch: u64, lim: &Limits) -> Result<(), Error> {
    src.validate()?;
    dst.validate()?;
    let pixels = src.pixels().max(dst.pixels());
    if pixels > lim.max_pixels {
        return Err(Error::PixelLimit {
            got: pixels,
            max: lim.max_pixels,
        });
    }
    if scratch > lim.max_alloc as u64 {
        return Err(Error::AllocLimit {
            got: scratch,
            max: lim.max_alloc as u64,
        });
    }
    let out = dst.pixels() * dst.channels as u64;
    if out > lim.max_output {
        return Err(Error::OutputLimit {
            got: out,
            max: lim.max_output,
        });
    }
    Ok(())
}

/// Halves `src` exactly: each output sample is the rounded mean `(sum+2)>>2` of a 2x2 block.
/// `dst` must be `src.w/2` by `src.h/2` with the same channels; an odd last row or column is
/// dropped.
///
/// Limits: `max_pixels` (the larger plane), `max_output` (the bytes of `dst`). It allocates
/// nothing, so `max_alloc` does not apply. It splits whole destination rows over up to
/// `max_threads` threads (1 touches no pool), with the same output for every count.
///
/// # Errors
///
/// - [`Error::Invalid`]: a bad plane (zero size, channels not 1 or 3, stride below the row, a
///   buffer too short), `src` under 2x2, or `dst` not `src.w/2` by `src.h/2` with the same channels.
/// - [`Error::PixelLimit`]: a plane has more than `max_pixels` pixels.
/// - [`Error::OutputLimit`]: `dst` is more than `max_output` bytes.
/// - [`Error::Cancelled`]: the flag was set; `dst` may be partly written.
///
/// ```
/// # use raster::{Limits, Plane, PlaneMut};
/// # use std::sync::atomic::AtomicBool;
/// let src = Plane { pix: &[10, 20, 30, 40, 50, 60, 70, 80], w: 4, h: 2, stride: 4, channels: 1 };
/// let mut out = [0u8; 2];
/// let mut dst = PlaneMut { pix: &mut out, w: 2, h: 1, stride: 2, channels: 1 };
/// raster::scale2to1(&mut dst, &src, &Limits::default(), &AtomicBool::new(false)).unwrap();
/// assert_eq!(out, [35, 55]);
/// ```
pub fn scale2to1(
    dst: &mut PlaneMut,
    src: &Plane,
    lim: &Limits,
    cancel: &AtomicBool,
) -> Result<(), Error> {
    scale2to1_with(dst, src, lim, cancel, &Budget::new(lim))
}

fn scale2to1_with(
    dst: &mut PlaneMut,
    src: &Plane,
    lim: &Limits,
    cancel: &AtomicBool,
    bud: &Budget,
) -> Result<(), Error> {
    check(dst.dims(), src.dims(), 0, lim)?;
    if src.w < 2
        || src.h < 2
        || dst.channels != src.channels
        || dst.w != src.w / 2
        || dst.h != src.h / 2
    {
        return Err(Error::Invalid("dst must be src/2 with equal channels"));
    }
    scale::box2to1(
        dst.pix,
        dst.stride,
        src.pix,
        src.stride,
        (dst.w, dst.h, dst.channels),
        Ctx { bud, cancel },
    )
}

/// Converts RGB to grey with integer BT.601 weights, `Y = (38R + 75G + 15B + 64) >> 7`. `src`
/// must have 3 channels, `dst` 1, and both the same size.
///
/// Limits: `max_pixels`, `max_output` (the bytes of `dst`). It allocates nothing, so `max_alloc`
/// does not apply. It splits whole rows over up to `max_threads` threads (1 touches no pool),
/// with the same output for every count.
///
/// # Errors
///
/// - [`Error::Invalid`]: a bad plane, `src` not 3 channels, `dst` not 1, or sizes that differ.
/// - [`Error::PixelLimit`]: a plane has more than `max_pixels` pixels.
/// - [`Error::OutputLimit`]: `dst` is more than `max_output` bytes.
/// - [`Error::Cancelled`]: the flag was set; `dst` may be partly written.
pub fn luma(
    dst: &mut PlaneMut,
    src: &Plane,
    lim: &Limits,
    cancel: &AtomicBool,
) -> Result<(), Error> {
    luma_with(dst, src, lim, cancel, &Budget::new(lim))
}

fn luma_with(
    dst: &mut PlaneMut,
    src: &Plane,
    lim: &Limits,
    cancel: &AtomicBool,
    bud: &Budget,
) -> Result<(), Error> {
    check(dst.dims(), src.dims(), 0, lim)?;
    if src.channels != 3 || dst.channels != 1 || dst.w != src.w || dst.h != src.h {
        return Err(Error::Invalid(
            "luma needs RGB src and grey dst of equal size",
        ));
    }
    luma::plane(
        dst.pix,
        dst.stride,
        src.pix,
        src.stride,
        (src.w, src.h),
        Ctx { bud, cancel },
    )
}

/// Rotates `src` a quarter turn (clockwise or not) into `dst`, which must be `src.h` by `src.w`
/// with the same channels.
///
/// Limits: `max_pixels`, `max_output` (the bytes of `dst`). It allocates nothing, so `max_alloc`
/// does not apply. It splits whole 16-row tile bands of `dst` over up to `max_threads` threads
/// (1 touches no pool), with the same output for every count; the flag is checked per tile band.
///
/// # Errors
///
/// - [`Error::Invalid`]: a bad plane, a channel mismatch, or `dst` not `src` transposed.
/// - [`Error::PixelLimit`]: a plane has more than `max_pixels` pixels.
/// - [`Error::OutputLimit`]: `dst` is more than `max_output` bytes.
/// - [`Error::Cancelled`]: the flag was set; `dst` may be partly written.
pub fn rotate90(
    dst: &mut PlaneMut,
    src: &Plane,
    clockwise: bool,
    lim: &Limits,
    cancel: &AtomicBool,
) -> Result<(), Error> {
    rotate90_with(dst, src, clockwise, lim, cancel, &Budget::new(lim))
}

fn rotate90_with(
    dst: &mut PlaneMut,
    src: &Plane,
    clockwise: bool,
    lim: &Limits,
    cancel: &AtomicBool,
    bud: &Budget,
) -> Result<(), Error> {
    check(dst.dims(), src.dims(), 0, lim)?;
    if dst.channels != src.channels || dst.w != src.h || dst.h != src.w {
        return Err(Error::Invalid(
            "dst must be src transposed with equal channels",
        ));
    }
    rotate::quarter(
        dst.pix,
        dst.stride,
        src.pix,
        src.stride,
        (src.w, src.h, src.channels),
        clockwise,
        Ctx { bud, cancel },
    )
}

/// Scales `src` to `dst`'s size with a separable Catmull-Rom filter, Q14 taps and i32
/// accumulation, horizontal pass first. The filter widens when downscaling. It allocates an
/// intermediate plane, the tap tables and an accumulator row for each band of the vertical pass,
/// charged against `Limits::max_alloc` first.
///
/// Limits: `max_pixels`, `max_output` (the bytes of `dst`), `max_alloc` (the peak working memory
/// of the call across all threads; Go's up-front check of the single-thread need also applies).
/// The horizontal pass splits rows of the intermediate plane and the vertical pass rows of `dst`
/// over up to `max_threads` threads (1 touches no pool), with the same output for every count.
///
/// # Errors
///
/// - [`Error::Invalid`]: a bad plane, a channel mismatch, or a dimension above `i32::MAX`.
/// - [`Error::PixelLimit`]: a plane has more than `max_pixels` pixels.
/// - [`Error::AllocLimit`]: the need, or a charge on the ledger, would pass `max_alloc`.
/// - [`Error::OutputLimit`]: `dst` is more than `max_output` bytes.
/// - [`Error::OutOfMemory`]: the allocator refused a size within the limits.
/// - [`Error::Cancelled`]: the flag was set; `dst` may be partly written.
///
/// It is [`Resampler::default().resample`](Resampler::resample): a call that keeps nothing. A
/// worker that resamples tile after tile should keep a [`Resampler`].
pub fn resample(
    dst: &mut PlaneMut,
    src: &Plane,
    lim: &Limits,
    cancel: &AtomicBool,
) -> Result<(), Error> {
    Resampler::default().resample(dst, src, lim, cancel)
}

impl Resampler {
    /// [`resample`] on the buffers this `Resampler` keeps. The output is byte-identical to
    /// [`resample`] for every input, `max_threads` and sequence of earlier calls on `self`.
    ///
    /// Each call first charges the kept buffers at their capacity, and whatever it must grow,
    /// against `Limits::max_alloc`, before it grows anything; so `max_alloc` is still the peak
    /// of the call. If the kept buffers would take the call over it, the ones bigger than the
    /// call needs are freed first, so a call that passes as [`resample`] passes here. Go's
    /// up-front check of the single-thread need is the same. After [`Error::AllocLimit`] the
    /// `Resampler` holds nothing; after any other error it stays usable.
    ///
    /// With `max_threads` 1 a call of a shape it has run before (the same source and destination
    /// size) allocates nothing, provided its `max_alloc` is no tighter, and its `max_threads` no
    /// lower, than the call that built the buffers (otherwise the call trims the buffers the
    /// ledger cannot hold, and allocates them again); a larger shape grows the buffers once. With more threads the call
    /// also makes a few small allocations in the pool (job boxes and the list of band results),
    /// and keeps its accumulator rows: at least one per band it ran, plus any an earlier call left.
    ///
    /// Limits: as [`resample`].
    ///
    /// # Errors
    ///
    /// - [`Error::Invalid`]: a bad plane, a channel mismatch, or a dimension above `i32::MAX`.
    /// - [`Error::PixelLimit`]: a plane has more than `max_pixels` pixels.
    /// - [`Error::AllocLimit`]: the need, or a charge on the ledger, would pass `max_alloc`.
    /// - [`Error::OutputLimit`]: `dst` is more than `max_output` bytes.
    /// - [`Error::OutOfMemory`]: the allocator refused a size within the limits.
    /// - [`Error::Cancelled`]: the flag was set; `dst` may be partly written.
    ///
    /// ```
    /// use raster::{Limits, Plane, PlaneMut, Resampler};
    /// use std::sync::atomic::AtomicBool;
    ///
    /// let limits = Limits { max_threads: 1, max_alloc: 64 << 20, ..Limits::default() };
    /// let cancel = AtomicBool::new(false);
    /// let src_pix: Vec<u8> = (0..16 * 16).map(|i| (i * 7 % 251) as u8).collect();
    /// let src = Plane { pix: &src_pix, w: 16, h: 16, stride: 16, channels: 1 };
    /// let mut rs = Resampler::default();
    /// let mut out = [0u8; 8 * 8];
    /// // The second call reuses everything the first one built.
    /// for _ in 0..2 {
    ///     let mut dst = PlaneMut { pix: &mut out, w: 8, h: 8, stride: 8, channels: 1 };
    ///     rs.resample(&mut dst, &src, &limits, &cancel).unwrap();
    /// }
    /// let mut want = [0u8; 8 * 8];
    /// let mut dst = PlaneMut { pix: &mut want, w: 8, h: 8, stride: 8, channels: 1 };
    /// raster::resample(&mut dst, &src, &limits, &cancel).unwrap();
    /// assert_eq!(out, want);
    /// ```
    pub fn resample(
        &mut self,
        dst: &mut PlaneMut,
        src: &Plane,
        lim: &Limits,
        cancel: &AtomicBool,
    ) -> Result<(), Error> {
        resample_with(dst, src, lim, cancel, &Budget::new(lim), self)
    }
}

fn resample_with(
    dst: &mut PlaneMut,
    src: &Plane,
    lim: &Limits,
    cancel: &AtomicBool,
    bud: &Budget,
    rs: &mut Resampler,
) -> Result<(), Error> {
    src.dims().validate()?;
    dst.dims().validate()?;
    if dst.channels != src.channels {
        return Err(Error::Invalid("channel mismatch"));
    }
    let scratch = resample::scratch_bytes(src.w, src.h, dst.w, dst.h, src.channels);
    let checked = check(dst.dims(), src.dims(), scratch, lim);
    if matches!(checked, Err(Error::AllocLimit { .. })) {
        *rs = Resampler::default(); // an over-limit call keeps nothing
    }
    checked?;
    // The tap tables index with i32 (as Go does); Go wraps past 2^31, this refuses. After
    // `check`, so an input Go refuses on a limit gets the same class here.
    if [src.w, src.h, dst.w, dst.h]
        .iter()
        .any(|&n| n > i32::MAX as usize)
    {
        return Err(Error::Invalid("dimension above i32::MAX"));
    }
    rs.run(
        (dst.pix, dst.stride),
        (dst.w, dst.h),
        (src.pix, src.stride),
        (src.w, src.h, src.channels),
        Ctx { bud, cancel },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNSET: u8 = 0xEE;

    /// Runs `f` with the `n`th cancel check failing; returns its result.
    fn cancel_at<T>(n: usize, f: impl FnOnce() -> T) -> T {
        CANCEL_AT.with(|c| c.set(n));
        let r = f();
        CANCEL_AT.with(|c| c.set(0));
        r
    }

    fn resample_rows(sh: usize, n: usize) -> (Result<(), Error>, Vec<u8>, Vec<u8>) {
        let (sw, dw, dh) = (40, 20, 20);
        let src: Vec<u8> = (0..sw * sh).map(|i| (i * 7 % 251) as u8).collect();
        let s = Plane {
            pix: &src,
            w: sw,
            h: sh,
            stride: sw,
            channels: 1,
        };
        let run = |at| {
            let mut out = vec![UNSET; dw * dh];
            let mut d = PlaneMut {
                pix: &mut out,
                w: dw,
                h: dh,
                stride: dw,
                channels: 1,
            };
            let r = cancel_at(at, || {
                resample(&mut d, &s, &Limits::default(), &AtomicBool::new(false))
            });
            (r, out)
        };
        let (_, full) = run(0);
        let (r, part) = run(n);
        (r, part, full)
    }

    /// Go checks ctx per row in both passes: a cancel in the second pass keeps the rows done.
    #[test]
    fn resample_cancel_mid_call() {
        let sh = 40;
        // checks: 1 before the width taps, 2 before the height taps, then sh rows, then dh rows
        for (n, rows) in [
            (1, 0),
            (2, 0),
            (3, 0),
            (2 + sh, 0),
            (3 + sh, 0),
            (3 + sh + 5, 5),
        ] {
            let (r, part, full) = resample_rows(sh, n);
            assert!(matches!(r, Err(Error::Cancelled)), "n={n}");
            assert_eq!(part[..rows * 20], full[..rows * 20], "n={n}");
            assert!(part[rows * 20..].iter().all(|&b| b == UNSET), "n={n}");
        }
    }

    /// Rotate checks per tile row (16 source columns): cancel before the third keeps 32 rows.
    #[test]
    fn rotate_cancel_mid_call() {
        let (w, h) = (40, 20); // dst is 20 wide, 40 high: 3 tile rows
        let src: Vec<u8> = (0..w * h).map(|i| (i * 11 % 253) as u8).collect();
        let s = Plane {
            pix: &src,
            w,
            h,
            stride: w,
            channels: 1,
        };
        let run = |at| {
            let mut out = vec![UNSET; w * h];
            let mut d = PlaneMut {
                pix: &mut out,
                w: h,
                h: w,
                stride: h,
                channels: 1,
            };
            let r = cancel_at(at, || {
                rotate90(
                    &mut d,
                    &s,
                    true,
                    &Limits::default(),
                    &AtomicBool::new(false),
                )
            });
            (r, out)
        };
        let (_, full) = run(0);
        let (r, part) = run(3);
        assert!(matches!(r, Err(Error::Cancelled)));
        assert_eq!(part[..32 * h], full[..32 * h]);
        assert!(part[32 * h..].iter().all(|&b| b == UNSET));
    }

    fn pattern(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i * 7 + i / 13) as u8).collect()
    }

    /// `run` at 1 thread, then at other thread counts: the same result and bytes, never more
    /// threads than `max_threads`, and exactly 1 for 1.
    fn threads_agree(name: &str, run: impl Fn(&Limits, &Budget) -> (Result<(), Error>, Vec<u8>)) {
        let l1 = Limits {
            max_threads: 1,
            ..Limits::default()
        };
        let b1 = Budget::new(&l1);
        let (r, want) = run(&l1, &b1);
        assert_eq!(r, Ok(()), "{name}");
        assert_eq!(b1.peak_threads(), 1, "{name}");
        for t in [0usize, 2, 3, 8] {
            let l = Limits {
                max_threads: t,
                ..Limits::default()
            };
            let b = Budget::new(&l);
            let (r, got) = run(&l, &b);
            assert_eq!(r, Ok(()), "{name} threads {t}");
            assert!(got == want, "{name} threads {t}: bytes differ");
            assert!(b.peak_threads() <= t.max(1), "{name} threads {t}");
            assert_eq!(b.threads(), 1, "{name} threads {t}");
        }
    }

    /// A destination with `pad` spare bytes per row and `tail` spare bytes after the last.
    fn dst_buf(w: usize, h: usize, ch: usize, pad: usize, tail: usize) -> (Vec<u8>, usize) {
        let stride = w * ch + pad;
        (vec![UNSET; (h - 1) * stride + w * ch + tail], stride)
    }

    #[test]
    fn scale_luma_rotate_same_for_every_thread_count() {
        let (w, h) = (2400, 1800); // big enough to split on the SIMD paths too
        let cancel = AtomicBool::new(false);
        for ch in [1usize, 3] {
            let pix = pattern(w * h * ch);
            let s = Plane {
                pix: &pix,
                w,
                h,
                stride: w * ch,
                channels: ch,
            };
            threads_agree(&format!("scale ch {ch}"), |l, b| {
                let (mut out, stride) = dst_buf(w / 2, h / 2, ch, 5, 9);
                let mut d = PlaneMut {
                    pix: &mut out,
                    w: w / 2,
                    h: h / 2,
                    stride,
                    channels: ch,
                };
                let r = scale2to1_with(&mut d, &s, l, &cancel, b);
                (r, out)
            });
            for cw in [true, false] {
                threads_agree(&format!("rotate ch {ch} cw {cw}"), |l, b| {
                    let (mut out, stride) = dst_buf(h, w, ch, 3, 7);
                    let mut d = PlaneMut {
                        pix: &mut out,
                        w: h,
                        h: w,
                        stride,
                        channels: ch,
                    };
                    let r = rotate90_with(&mut d, &s, cw, l, &cancel, b);
                    (r, out)
                });
            }
            if ch == 3 {
                threads_agree("luma", |l, b| {
                    let (mut out, stride) = dst_buf(w, h, 1, 4, 11);
                    let mut d = PlaneMut {
                        pix: &mut out,
                        w,
                        h,
                        stride,
                        channels: 1,
                    };
                    let r = luma_with(&mut d, &s, l, &cancel, b);
                    (r, out)
                });
            }
        }
    }

    /// Resample gives the same bytes at every thread count and `max_alloc`; the ledger peak stays
    /// within `max_alloc`; the single-thread need passes at every thread count (the band count
    /// shrinks to fit) and one byte less is refused.
    #[test]
    fn resample_same_for_every_thread_count_and_alloc() {
        let (sw, sh, dw, dh) = (1000, 800, 700, 500);
        let cancel = AtomicBool::new(false);
        for ch in [1usize, 3] {
            let pix = pattern(sw * sh * ch);
            let s = Plane {
                pix: &pix,
                w: sw,
                h: sh,
                stride: sw * ch,
                channels: ch,
            };
            let need = resample::scratch_bytes(sw, sh, dw, dh, ch);
            let acc = (dw * ch * 4) as u64;
            let run = |threads: usize, max_alloc: u64| {
                let l = Limits {
                    max_threads: threads,
                    max_alloc: max_alloc as usize,
                    ..Limits::default()
                };
                let b = Budget::new(&l);
                let (mut out, stride) = dst_buf(dw, dh, ch, 2, 5);
                let mut d = PlaneMut {
                    pix: &mut out,
                    w: dw,
                    h: dh,
                    stride,
                    channels: ch,
                };
                let r = resample_with(&mut d, &s, &l, &cancel, &b, &mut Resampler::default());
                (r, out, b)
            };
            let (r, want, b1) = run(1, need);
            assert_eq!(r, Ok(()), "ch {ch}");
            assert_eq!(b1.peak_threads(), 1);
            assert!(b1.peak() <= need, "ch {ch}: peak {} need {need}", b1.peak());
            assert_eq!(b1.held(), 0);
            for t in [1usize, 2, 3, 8] {
                for alloc in [need, need + acc, need + 3 * acc, 1 << 28] {
                    let (r, got, b) = run(t, alloc);
                    let at = format!("ch {ch} threads {t} alloc {alloc}");
                    assert_eq!(r, Ok(()), "{at}");
                    assert!(got == want, "{at}: bytes differ");
                    assert!(b.peak() <= alloc, "{at}: peak {}", b.peak());
                    assert!(b.peak_threads() <= t, "{at}");
                    assert_eq!(b.held(), 0, "{at}");
                    assert_eq!(b.threads(), 1, "{at}");
                }
                let (r, _, b) = run(t, need - 1);
                assert_eq!(
                    r,
                    Err(Error::AllocLimit {
                        got: need,
                        max: need - 1
                    }),
                    "ch {ch} threads {t}"
                );
                assert_eq!(b.peak(), 0);
            }
        }
    }

    /// One `Resampler` that first ran a big call at 4 threads, then small calls at the tightest
    /// `max_alloc` (the need) and one accumulator row above it, at 1 and 4 threads: the ledger's
    /// peak stays within `max_alloc` (the kept buffers are charged or trimmed), it is empty after
    /// each call, and the output equals the free function's.
    #[test]
    fn reused_resampler_peak_within_max_alloc() {
        let cancel = AtomicBool::new(false);
        let pix = pattern(1000 * 800);
        let big = Plane {
            pix: &pix,
            w: 1000,
            h: 800,
            stride: 1000,
            channels: 1,
        };
        let (sw, sh, dw, dh) = (600, 500, 400, 300);
        let small = Plane {
            pix: &pix[..sw * sh],
            w: sw,
            h: sh,
            stride: sw,
            channels: 1,
        };
        let need = resample::scratch_bytes(sw, sh, dw, dh, 1);
        let acc = (dw * 4) as u64;
        let mut rs = Resampler::default();
        let l = Limits {
            max_threads: 4,
            ..Limits::default()
        };
        let mut out = vec![0u8; 700 * 500];
        let mut d = PlaneMut {
            pix: &mut out,
            w: 700,
            h: 500,
            stride: 700,
            channels: 1,
        };
        let b = Budget::new(&l);
        assert_eq!(
            resample_with(&mut d, &big, &l, &cancel, &b, &mut rs),
            Ok(())
        );
        assert_eq!(b.held(), 0);
        for threads in [1usize, 4, 1] {
            for max_alloc in [need, need + acc] {
                let l = Limits {
                    max_threads: threads,
                    max_alloc: max_alloc as usize,
                    ..Limits::default()
                };
                let mut want = vec![0u8; dw * dh];
                let mut got = vec![0u8; dw * dh];
                let mut want_d = PlaneMut {
                    pix: &mut want,
                    w: dw,
                    h: dh,
                    stride: dw,
                    channels: 1,
                };
                assert_eq!(resample(&mut want_d, &small, &l, &cancel), Ok(()));
                let mut got_d = PlaneMut {
                    pix: &mut got,
                    w: dw,
                    h: dh,
                    stride: dw,
                    channels: 1,
                };
                let b = Budget::new(&l);
                let r = resample_with(&mut got_d, &small, &l, &cancel, &b, &mut rs);
                let at = format!("threads {threads} max_alloc {max_alloc}");
                assert_eq!(r, Ok(()), "{at}");
                assert!(got == want, "{at}: bytes differ");
                assert!(b.peak() <= max_alloc, "{at}: peak {}", b.peak());
                assert_eq!(b.held(), 0, "{at}");
            }
        }
    }

    /// The ledger itself refuses a charge over `max_alloc`, with the up-front check bypassed.
    #[test]
    fn resample_ledger_refuses_past_max_alloc() {
        let (sw, sh, dw, dh) = (200, 100, 100, 50);
        let pix = pattern(sw * sh);
        let l = Limits {
            max_alloc: 64,
            max_threads: 1,
            ..Limits::default()
        };
        let mut out = vec![0u8; dw * dh];
        let r = Resampler::default().run(
            (&mut out, dw),
            (dw, dh),
            (&pix, sw),
            (sw, sh, 1),
            Ctx {
                bud: &Budget::new(&l),
                cancel: &AtomicBool::new(false),
            },
        );
        assert!(matches!(r, Err(Error::AllocLimit { max: 64, .. })), "{r:?}");
    }
}
