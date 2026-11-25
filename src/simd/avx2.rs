//! AVX2 kernels, 32 box outputs or 16 luma pixels per step.
//!
//! The kernels are safe `target_feature` fns with no `unsafe` in them. `unsafe` is only in the
//! dispatch fns (`box_gray`, `luma`), which call a kernel once `detect` has checked AVX2, and in
//! the load/store helpers, whose array-reference parameters prove the length.

use super::Kernels;
use std::arch::x86_64::*;

pub(super) fn detect() -> Option<Kernels> {
    is_x86_feature_detected!("avx2").then_some(Kernels {
        name: "avx2",
        box_gray,
        luma,
        box_rgb: None,
        tile: None,
        vert: None,
    })
}

fn ld128(s: &[u8; 16]) -> __m128i {
    // SAFETY: the array type proves `s` is 16 bytes; the load is unaligned. SSE2 is baseline on
    // x86_64, so calling the intrinsic without `target_feature` is sound.
    unsafe { _mm_loadu_si128(s.as_ptr().cast()) }
}

#[target_feature(enable = "avx2")]
fn ld256(s: &[u8; 32]) -> __m256i {
    // SAFETY: the array type proves `s` is 32 bytes; the load is unaligned.
    unsafe { _mm256_loadu_si256(s.as_ptr().cast()) }
}

fn st128(d: &mut [u8; 16], v: __m128i) {
    // SAFETY: the array type proves `d` is 16 bytes; the store is unaligned. SSE2 is baseline on
    // x86_64.
    unsafe { _mm_storeu_si128(d.as_mut_ptr().cast(), v) }
}

#[target_feature(enable = "avx2")]
fn st256(d: &mut [u8; 32], v: __m256i) {
    // SAFETY: the array type proves `d` is 32 bytes; the store is unaligned.
    unsafe { _mm256_storeu_si256(d.as_mut_ptr().cast(), v) }
}

/// 16 bytes at the constant offset `O` of a 52-byte luma window; the check folds away.
fn sub<const O: usize>(b: &[u8; 52]) -> Option<&[u8; 16]> {
    b.get(O..)?.first_chunk()
}

/// Also the AVX-512 tier's grey box (`avx512::detect` checks AVX2 too).
pub(super) fn box_gray(dst: &mut [u8], r0: &[u8], r1: &[u8]) -> usize {
    let n = dst.len() & !31;
    // SAFETY: AVX2 is present if this is called through the `box_gray` field of a `Kernels` that
    // `detect` here or `avx512::detect` built, which both checked AVX2 before building it. The fn is
    // `pub(super)` and safe, so the compiler does not enforce that: the only references to it are
    // those two fields (the unit tests and `scale` call it through `Kernels`), and a new caller
    // must keep to that.
    unsafe { box_kernel(&mut dst[..n], &r0[..2 * n], &r1[..2 * n]) };
    n
}

/// (a+b+c+d+2)>>2: VPMADDUBSW with ones sums byte pairs, VPAVGB would round up so the rounding
/// is explicit.
#[target_feature(enable = "avx2")]
fn box_kernel(dst: &mut [u8], r0: &[u8], r1: &[u8]) {
    let ones = _mm256_set1_epi8(1);
    let two = _mm256_set1_epi16(2);
    let rows = r0
        .as_chunks::<32>()
        .0
        .as_chunks::<2>()
        .0
        .iter()
        .zip(r1.as_chunks::<32>().0.as_chunks::<2>().0);
    for (d, ([p0, p1], [q0, q1])) in dst.as_chunks_mut::<32>().0.iter_mut().zip(rows) {
        let a0 = _mm256_maddubs_epi16(ld256(p0), ones);
        let a1 = _mm256_maddubs_epi16(ld256(p1), ones);
        let b0 = _mm256_maddubs_epi16(ld256(q0), ones);
        let b1 = _mm256_maddubs_epi16(ld256(q1), ones);
        let s0 = _mm256_srli_epi16::<2>(_mm256_add_epi16(_mm256_add_epi16(a0, b0), two));
        let s1 = _mm256_srli_epi16::<2>(_mm256_add_epi16(_mm256_add_epi16(a1, b1), two));
        st256(
            d,
            _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(s0, s1)),
        );
    }
}

fn luma(dst: &mut [u8], src: &[u8]) -> usize {
    let n = dst.len().saturating_sub(2) / 16 * 16; // the last load reads 4 bytes past 3n
    if n == 0 {
        return 0;
    }
    // SAFETY: only reachable through `detect`, which checked AVX2 before this fn pointer could be
    // handed out.
    unsafe { luma_kernel(&mut dst[..n], &src[..3 * n + 4]) };
    n
}

/// Y = (38R+75G+15B+64)>>7, 16 pixels per step: VPSHUFB gathers R,G of 4 pixels and B,0 of the
/// same into each 128-bit lane, VPMADDUBSW weighs them. The loads at o, o+12, o+24 and o+36 need
/// 52 bytes from o = 48 * i.
#[target_feature(enable = "avx2")]
fn luma_kernel(dst: &mut [u8], src: &[u8]) {
    let shuf = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, 1, 3, 4, 6, 7, 9, 10, 2, -128, 5, -128, 8, -128, 11, -128,
    ));
    let wts = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        38, 75, 38, 75, 38, 75, 38, 75, 15, 0, 15, 0, 15, 0, 15, 0,
    ));
    let rnd = _mm256_set1_epi16(64);
    for (i, d) in dst.as_chunks_mut::<16>().0.iter_mut().enumerate() {
        let o = 48 * i;
        // One bounds check per step: the 52-byte window, then constant offsets inside it.
        let Some(blk) = src.get(o..).and_then(<[u8]>::first_chunk::<52>) else {
            break;
        };
        let (Some(s0), Some(s1), Some(s2), Some(s3)) = (
            sub::<0>(blk),
            sub::<12>(blk),
            sub::<24>(blk),
            sub::<36>(blk),
        ) else {
            break;
        };
        let y0 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(ld128(s0)), ld128(s2));
        let y1 = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(ld128(s1)), ld128(s3));
        let y0 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(y0, shuf), wts);
        let y1 = _mm256_maddubs_epi16(_mm256_shuffle_epi8(y1, shuf), wts);
        let y0 = _mm256_add_epi16(y0, _mm256_srli_si256::<8>(y0));
        let y1 = _mm256_add_epi16(y1, _mm256_srli_si256::<8>(y1));
        let y = _mm256_srli_epi16::<7>(_mm256_add_epi16(_mm256_unpacklo_epi64(y0, y1), rnd));
        let y = _mm256_permute4x64_epi64::<0x08>(_mm256_packus_epi16(y, y));
        st128(d, _mm256_castsi256_si128(y));
    }
}
