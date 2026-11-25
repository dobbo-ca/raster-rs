//! SIMD tiers: AVX-512 and AVX2 on x86_64, NEON on aarch64, each beside the scalar reference in
//! `scale`, `luma`, `rotate` and `resample`. AVX2 and NEON have the grey 2:1 box and RGB to luma
//! (the two kernels Go has assembly for); AVX-512 adds the RGB 2:1 box, the 16x16 rotate tile
//! (grey) and the vertical resample pass. The tier is detected once, best first; `RASTER_SIMD=scalar`
//! forces scalar and `RASTER_SIMD=avx2` caps x86_64 at AVX2 (any other value detects normally).
//! A kernel a tier lacks is `None` and the scalar code runs it. Scalar is the oracle: every tier
//! is bit-exact with it.

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2;
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx512;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon;

use std::sync::OnceLock;

/// `(dst, r0, r1) -> outputs done`: a 2:1 box row.
type BoxRow = fn(&mut [u8], &[u8], &[u8]) -> usize;
/// `(dst, dst_stride, src, src_stride, clockwise) -> done`: one rotate tile.
type TileFn = fn(&mut [u8], usize, &[u8], usize, bool) -> bool;
/// `(out, mid, mrow, first, ws) -> outputs done`: one vertical resample row.
type VertRow = fn(&mut [u8], &[u8], usize, usize, &[i16]) -> usize;

/// The kernels of one tier. Each does a whole number of lanes from the start of the row and
/// returns how many outputs it completed (see the field docs for `box_rgb`'s 2-byte spill and
/// for `tile`); the caller finishes the tail with the scalar code.
#[derive(Clone, Copy)]
pub(crate) struct Kernels {
    name: &'static str,
    /// `(dst, r0, r1)`: `dst.len()` outputs from `2 * dst.len()` bytes of each row.
    pub box_gray: BoxRow,
    /// `(dst, src)`: `dst.len()` pixels from `3 * dst.len()` bytes.
    pub luma: fn(&mut [u8], &[u8]) -> usize,
    /// `(dst, r0, r1)`: `dst.len()` RGB samples from `2 * dst.len()` bytes of each row. It may
    /// write up to 2 bytes past the count it returns (the scalar tail overwrites them) and
    /// returns a multiple of 3.
    pub box_rgb: Option<BoxRow>,
    /// `(dst, dst_stride, src, src_stride, clockwise)`: one 16x16 grey rotate tile. `src` starts
    /// at the top-left of the 16x16 source block, `dst` at the top-left of the 16x16 destination
    /// tile. False (nothing promised written) if either slice is too short for the tile.
    pub tile: Option<TileFn>,
    /// `(out, mid, mrow, first, ws)`: the vertical resample pass of one row. `out[x]` is
    /// `clamp8(ONE/2 + sum_k ws[k] * mid[(first + k) * mrow + x])`; returns how many `x` it did.
    pub vert: Option<VertRow>,
}

/// The best tier this host has, or AVX2 at most if `cap_avx2`.
#[cfg(target_arch = "x86_64")]
fn detect(cap_avx2: bool) -> Option<Kernels> {
    if !cap_avx2 && let Some(k) = avx512::detect() {
        return Some(k);
    }
    avx2::detect()
}
#[cfg(target_arch = "aarch64")]
fn detect(_cap_avx2: bool) -> Option<Kernels> {
    neon::detect()
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect(_cap_avx2: bool) -> Option<Kernels> {
    None
}

/// Fuzz hook (`--cfg fuzzing` only): while set, every kernel is scalar, so a target can compare
/// the tier with scalar in one process.
#[cfg(fuzzing)]
static FORCE_SCALAR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Fuzz hook: forces the scalar code while `on`. Not for callers.
#[cfg(fuzzing)]
pub fn force_scalar(on: bool) {
    FORCE_SCALAR.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn kernels() -> Option<&'static Kernels> {
    #[cfg(fuzzing)]
    {
        if FORCE_SCALAR.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
    }
    static K: OnceLock<Option<Kernels>> = OnceLock::new();
    K.get_or_init(|| match std::env::var_os("RASTER_SIMD") {
        Some(v) if v == "scalar" => None,
        Some(v) if v == "avx2" => detect(true),
        _ => detect(false),
    })
    .as_ref()
}

/// The tier the kernels dispatch to on this host: "avx512", "avx2", "neon" or "generic"
/// (scalar). `RASTER_SIMD=scalar` or `avx2` lowers it. A tier may lack a kernel (AVX2 and NEON
/// have only the grey `scale2to1` and `luma`); that kernel then runs the scalar code.
///
/// It takes no `Limits`: it allocates nothing and spawns no thread.
///
/// # Errors
///
/// None: the query cannot fail.
pub fn tier() -> &'static str {
    kernels().map_or("generic", |k| k.name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{luma, scale};

    fn xorshift(state: &mut u64) -> u8 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 24) as u8
    }

    fn data(seed: u64, n: usize, fill: Option<u8>) -> Vec<u8> {
        let mut st = seed;
        (0..n)
            .map(|_| fill.unwrap_or_else(|| xorshift(&mut st)))
            .collect()
    }

    const SIZES: &[usize] = &[
        0, 1, 2, 15, 16, 17, 18, 31, 32, 33, 34, 47, 48, 49, 50, 63, 64, 65, 127, 128, 129, 200,
    ];
    /// RGB box output counts: around 32 + 30k, where the steps of AVX-512 change.
    const RGB_SIZES: &[usize] = &[
        0, 1, 2, 3, 29, 30, 31, 32, 33, 34, 35, 61, 62, 63, 64, 65, 66, 92, 93, 94, 95, 96, 97,
        200, 291, 300,
    ];

    /// Every tier this host has, best first. The CI hosts must have AVX2 (x86_64) or NEON
    /// (aarch64), and AVX-512 where `RASTER_EXPECT_AVX512=1` (set it only on a host with
    /// AVX-512).
    fn tiers() -> Vec<Kernels> {
        #[allow(unused_mut)]
        let mut v = Vec::new();
        #[cfg(target_arch = "x86_64")]
        {
            assert!(is_x86_feature_detected!("avx2"), "AVX2 expected on x86_64");
            let want512 = std::env::var_os("RASTER_EXPECT_AVX512").is_some_and(|v| v == "1");
            assert!(
                !want512 || avx512::detect().is_some(),
                "AVX-512 (F, BW, VBMI) expected: RASTER_EXPECT_AVX512=1"
            );
            v.extend(avx512::detect());
            v.extend(avx2::detect());
        }
        #[cfg(target_arch = "aarch64")]
        {
            assert!(
                std::arch::is_aarch64_feature_detected!("neon"),
                "NEON expected on aarch64"
            );
            v.extend(neon::detect());
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        assert!(!v.is_empty());
        v
    }

    #[derive(Clone, Copy)]
    enum Op {
        Box,
        Luma,
        Rgb,
    }

    /// Outputs a tier does in one call: whole lanes (AVX2 luma also keeps 2 pixels back for its
    /// last 4-byte over-read; AVX-512 RGB does steps of 30 and needs 2 spare outputs).
    fn lanes(k: &Kernels, op: Op, n: usize) -> usize {
        match (k.name, op) {
            ("avx512", Op::Box) => n & !31,
            ("avx512", Op::Luma) => n & !31,
            ("avx512", Op::Rgb) if n < 32 => 0,
            ("avx512", Op::Rgb) => (n - 32) / 30 * 30 + 30,
            ("avx2", Op::Luma) => n.saturating_sub(2) / 16 * 16,
            ("avx2", Op::Box) => n & !31,
            _ => n & !15,
        }
    }

    #[test]
    fn box_gray_matches_scalar() {
        for k in tiers() {
            for (seed, fill) in [(1, None), (2, None), (3, Some(0xff)), (4, Some(0))] {
                let (a, b) = (data(seed, 500, fill), data(seed + 9, 500, fill));
                for off in 0..4 {
                    for &dw in SIZES {
                        let (r0, r1) = (&a[off..off + 2 * dw], &b[off..off + 2 * dw]);
                        let mut got = vec![0u8; dw];
                        let done = (k.box_gray)(&mut got, r0, r1);
                        assert_eq!(done, lanes(&k, Op::Box, dw), "{} dw {dw}", k.name);
                        let mut want = vec![0u8; done];
                        scale::row(&mut want, &r0[..2 * done], &r1[..2 * done], 1);
                        assert_eq!(got[..done], want[..], "{} off {off} dw {dw}", k.name);
                        assert!(got[done..].iter().all(|&x| x == 0), "{} wrote past", k.name);
                    }
                }
            }
        }
    }

    #[test]
    fn box_rgb_matches_scalar() {
        for k in tiers() {
            let Some(f) = k.box_rgb else { continue };
            for (seed, fill) in [(11, None), (12, None), (13, Some(0xff)), (14, Some(0))] {
                let (a, b) = (data(seed, 700, fill), data(seed + 9, 700, fill));
                for off in 0..4 {
                    for &n in RGB_SIZES {
                        let (r0, r1) = (&a[off..off + 2 * n], &b[off..off + 2 * n]);
                        let mut got = vec![0u8; n];
                        let done = f(&mut got, r0, r1);
                        assert_eq!(done, lanes(&k, Op::Rgb, n), "{} n {n}", k.name);
                        assert_eq!(done % 3, 0);
                        let mut want = vec![0u8; done];
                        scale::row(&mut want, &r0[..2 * done], &r1[..2 * done], 3);
                        assert_eq!(got[..done], want[..], "{} off {off} n {n}", k.name);
                        // At most 2 bytes past `done`, which the scalar tail overwrites.
                        let slack = if done == 0 { 0 } else { 2 };
                        assert!(got[done + slack..].iter().all(|&x| x == 0));
                    }
                }
            }
        }
    }

    #[test]
    fn luma_matches_scalar() {
        for k in tiers() {
            for (seed, fill) in [(5, None), (6, None), (7, Some(0xff)), (8, Some(0))] {
                let a = data(seed, 700, fill);
                for off in 0..4 {
                    for &w in SIZES {
                        let src = &a[off..off + 3 * w];
                        let mut got = vec![0u8; w];
                        let done = (k.luma)(&mut got, src);
                        assert_eq!(done, lanes(&k, Op::Luma, w), "{} w {w}", k.name);
                        let mut want = vec![0u8; done];
                        luma::row(&mut want, &src[..3 * done]);
                        assert_eq!(got[..done], want[..], "{} off {off} w {w}", k.name);
                        assert!(got[done..].iter().all(|&x| x == 0), "{} wrote past", k.name);
                    }
                }
            }
        }
    }

    /// The rotate tile equals the definition (dst(t, u) is block(15-u, t) clockwise, block(u,
    /// 15-t) counter-clockwise), touches nothing else, and refuses a slice too short for it.
    #[test]
    fn tile_matches_definition() {
        for k in tiers() {
            let Some(f) = k.tile else { continue };
            for (ss, ds) in [(16usize, 16usize), (17, 19), (40, 33), (64, 16)] {
                let src = data(31 + ss as u64, 16 * ss + 16 + 8, None);
                for off in 0..4 {
                    for cw in [true, false] {
                        let mut got = vec![0xEEu8; 16 * ds + 16 + 8];
                        let mut want = got.clone();
                        for t in 0..16 {
                            for u in 0..16 {
                                let (r, c) = if cw { (15 - u, t) } else { (u, 15 - t) };
                                want[off + t * ds + u] = src[off + r * ss + c];
                            }
                        }
                        assert!(f(&mut got[off..], ds, &src[off..], ss, cw));
                        assert_eq!(got, want, "{} ss {ss} ds {ds} off {off} cw {cw}", k.name);
                    }
                }
                let mut got = vec![0u8; 16 * ds + 16];
                assert!(!f(&mut got, ds, &src[..15 * ss + 15], ss, true));
                assert!(!f(&mut got[..15 * ds + 15], ds, &src, ss, false));
            }
        }
    }

    #[test]
    fn vert_matches_scalar() {
        for k in tiers() {
            let Some(f) = k.vert else { continue };
            let mut st = 99u64;
            for (seed, fill) in [(21, None), (22, Some(0xff)), (23, Some(0))] {
                let buf = data(seed, 12 * 200 + 8, fill);
                for off in 0..3 {
                    let mid = &buf[off..];
                    for &mrow in SIZES {
                        for cnt in [0usize, 1, 2, 3, 4, 9, 10] {
                            let first = (off + cnt) % 3;
                            let ws: Vec<i16> = (0..cnt)
                                .map(|_| (i16::from(xorshift(&mut st)) - 40) * 100)
                                .collect();
                            let mut got = vec![0u8; mrow];
                            let done = f(&mut got, mid, mrow, first, &ws);
                            assert_eq!(done, mrow & !63, "{} mrow {mrow}", k.name);
                            for x in 0..done {
                                let sum: i32 = ws
                                    .iter()
                                    .enumerate()
                                    .map(|(j, &w)| {
                                        i32::from(w) * i32::from(mid[(first + j) * mrow + x])
                                    })
                                    .sum();
                                let want = ((8192 + sum) >> 14).clamp(0, 255) as u8;
                                assert_eq!(got[x], want, "{} mrow {mrow} cnt {cnt} x {x}", k.name);
                            }
                            assert!(got[done..].iter().all(|&b| b == 0), "{} wrote past", k.name);
                        }
                    }
                }
            }
        }
    }
}
