//! AVX-512 kernels (F, BW and VBMI; AVX2 for the 256-bit stores): 32 luma pixels, 30 RGB box
//! samples, one 16x16 grey rotate tile or 64 resample columns per step. The grey box uses the
//! AVX2 kernel (a 64-wide one was tried and was no faster; docs/safety.md).
//!
//! The kernels are safe `target_feature` fns with no `unsafe` in them. `unsafe` is only in the
//! dispatch fns (`luma`, `box_rgb`, `tile`, `vert`), which call a kernel once `detect`
//! has checked the features, and in the load/store helpers, whose array-reference parameters prove
//! the length. There are no masked loads or stores. Every kernel does whole steps and returns what
//! it did; the callers finish with the scalar code.

use super::Kernels;
use std::arch::x86_64::*;

pub(super) fn detect() -> Option<Kernels> {
    (is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vbmi"))
    .then_some(Kernels {
        name: "avx512",
        // The grey box stays on the AVX2 kernel: the 64-wide one was no faster (docs/safety.md).
        box_gray: super::avx2::box_gray,
        luma,
        box_rgb: Some(box_rgb),
        tile: Some(tile),
        vert: Some(vert),
    })
}

/// The `N` bytes of `s` from `i`, if there are that many.
fn at<const N: usize>(s: &[u8], i: usize) -> Option<&[u8; N]> {
    s.get(i..)?.first_chunk()
}

fn at_mut<const N: usize>(s: &mut [u8], i: usize) -> Option<&mut [u8; N]> {
    s.get_mut(i..)?.first_chunk_mut()
}

fn ld128(s: &[u8; 16]) -> __m128i {
    // SAFETY: the array type proves `s` is 16 bytes; the load is unaligned. SSE2 is baseline on
    // x86_64, so calling the intrinsic without `target_feature` is sound.
    unsafe { _mm_loadu_si128(s.as_ptr().cast()) }
}

fn st128(d: &mut [u8; 16], v: __m128i) {
    // SAFETY: the array type proves `d` is 16 bytes; the store is unaligned. SSE2 is baseline on
    // x86_64.
    unsafe { _mm_storeu_si128(d.as_mut_ptr().cast(), v) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn ld256(s: &[u8; 32]) -> __m256i {
    // SAFETY: the array type proves `s` is 32 bytes; the load is unaligned.
    unsafe { _mm256_loadu_si256(s.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn st256(d: &mut [u8; 32], v: __m256i) {
    // SAFETY: the array type proves `d` is 32 bytes; the store is unaligned.
    unsafe { _mm256_storeu_si256(d.as_mut_ptr().cast(), v) }
}

#[inline]
#[target_feature(enable = "avx512f")]
fn ld512(s: &[u8; 64]) -> __m512i {
    // SAFETY: the array type proves `s` is 64 bytes; the load is unaligned.
    unsafe { _mm512_loadu_si512(s.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx512f")]
fn st512(d: &mut [u8; 64], v: __m512i) {
    // SAFETY: the array type proves `d` is 64 bytes; the store is unaligned.
    unsafe { _mm512_storeu_si512(d.as_mut_ptr().cast(), v) }
}

// ---- luma -----------------------------------------------------------------------------------

/// Byte indices for VPERMT2B over a 96-byte block of 32 pixels (`a` = bytes 0..64, `b` = 64..96):
/// with `b_only` false, word i is (R_i, G_i); with it true, word i is (B_i, anything).
const fn luma_idx(b_only: bool) -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut i = 0;
    while i < 32 {
        if b_only {
            t[2 * i] = (3 * i + 2) as u8;
        } else {
            t[2 * i] = (3 * i) as u8;
            t[2 * i + 1] = (3 * i + 1) as u8;
        }
        i += 1;
    }
    t
}
const LUMA_RG: [u8; 64] = luma_idx(false);
const LUMA_B: [u8; 64] = luma_idx(true);

fn luma(dst: &mut [u8], src: &[u8]) -> usize {
    let n = dst.len() & !31;
    // SAFETY: only reachable through `detect`, which checked the AVX2, AVX-512 F, BW and VBMI
    // features before this fn pointer could be handed out.
    unsafe { luma_kernel(&mut dst[..n], &src[..3 * n]) };
    n
}

/// Y = (38R+75G+15B+64)>>7, 32 pixels (96 bytes, read exactly) per step: two VPERMT2B gather the
/// (R,G) and (B,x) words, VPMADDUBSW weighs them (at most 28815 + 3825 + 64, so 16 bits hold it),
/// VPMOVWB packs.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi")]
fn luma_kernel(dst: &mut [u8], src: &[u8]) {
    let rg = ld512(&LUMA_RG);
    let bb = ld512(&LUMA_B);
    let w_rg = _mm512_set1_epi16(i16::from_le_bytes([38, 75]));
    let w_b = _mm512_set1_epi16(15);
    let rnd = _mm512_set1_epi16(64);
    for (d, blk) in dst
        .as_chunks_mut::<32>()
        .0
        .iter_mut()
        .zip(src.as_chunks::<96>().0)
    {
        let (Some(lo), Some(hi)) = (blk.first_chunk::<64>(), blk.last_chunk::<32>()) else {
            break;
        };
        let a = ld512(lo);
        // The upper half of `b` is never selected: the indices stay below 96.
        let b = _mm512_castsi256_si512(ld256(hi));
        let y = _mm512_maddubs_epi16(_mm512_permutex2var_epi8(a, rg, b), w_rg);
        let y = _mm512_add_epi16(
            y,
            _mm512_maddubs_epi16(_mm512_permutex2var_epi8(a, bb, b), w_b),
        );
        let y = _mm512_srli_epi16::<7>(_mm512_add_epi16(y, rnd));
        st256(d, _mm512_cvtepi16_epi8(y));
    }
}

// ---- RGB 2:1 box ----------------------------------------------------------------------------

/// Byte indices for VPERMB over 64 bytes of one row: word j is (byte p, byte p+3) with
/// p = 6*(j/3) + j%3, the same channel of two adjacent pixels. Words 30 and 31 are not used (31
/// would need byte 64); the indices wrap into the register.
const fn rgb_idx() -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut j = 0;
    while j < 32 {
        let p = 2 * j - j % 3;
        t[2 * j] = (p & 63) as u8;
        t[2 * j + 1] = ((p + 3) & 63) as u8;
        j += 1;
    }
    t
}
const RGB_IDX: [u8; 64] = rgb_idx();

fn box_rgb(dst: &mut [u8], r0: &[u8], r1: &[u8]) -> usize {
    // A step writes 32 bytes at 30 * i (the last two are the next step's, or the scalar tail's)
    // and reads 64 bytes of each row at 60 * i.
    let steps = if dst.len() < 32 {
        0
    } else {
        (dst.len() - 32) / 30 + 1
    };
    let n = 30 * steps;
    if steps == 0 {
        return 0;
    }
    // SAFETY: only reachable through `detect`, which checked the AVX2, AVX-512 F, BW and VBMI
    // features before this fn pointer could be handed out.
    unsafe { box_rgb_kernel(&mut dst[..n + 2], &r0[..2 * n + 4], &r1[..2 * n + 4], steps) };
    n
}

/// (a+b+c+d+2)>>2 per sample, 30 samples (10 pixels, 60 bytes per row) per step: VPERMB lines up
/// byte pairs (a,b) and (c,d), VPMADDUBSW with ones sums each pair, VPMOVWB packs 32 words of
/// which the first 30 are kept.
#[target_feature(enable = "avx2,avx512f,avx512bw,avx512vbmi")]
fn box_rgb_kernel(dst: &mut [u8], r0: &[u8], r1: &[u8], steps: usize) {
    let ones = _mm512_set1_epi8(1);
    let two = _mm512_set1_epi16(2);
    let idx = ld512(&RGB_IDX);
    for i in 0..steps {
        let (Some(d), Some(p), Some(q)) = (
            at_mut::<32>(dst, 30 * i),
            at::<64>(r0, 60 * i),
            at::<64>(r1, 60 * i),
        ) else {
            break;
        };
        let a = _mm512_maddubs_epi16(_mm512_permutexvar_epi8(idx, ld512(p)), ones);
        let b = _mm512_maddubs_epi16(_mm512_permutexvar_epi8(idx, ld512(q)), ones);
        let s = _mm512_srli_epi16::<2>(_mm512_add_epi16(_mm512_add_epi16(a, b), two));
        st256(d, _mm512_cvtepi16_epi8(s));
    }
}

// ---- 16x16 rotate tile ----------------------------------------------------------------------

// The transpose works on 4 zmm registers Z_a, lane b of Z_a holding row 4a+b of the block, so
// Z_a[16b + 4e + f] = M[4a+b][4e+f]. Stage 1 (VPERMT2B on Z_{2p}, Z_{2p+1}) gives V_{p,h} with
// V[32e' + 8f + 4a' + b] = Z_{2p+a'}[16b + 4(2h+e') + f]; stage 2 (on V_{0,h}, V_{1,h}) gives
// Y_e[16f + 4a + b] = M[4a+b][4e+f], i.e. lane f of Y_e is column 4e+f of the block: one row of
// the transpose.

const fn tile_s1(h: usize) -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut q = 0;
    while q < 64 {
        let (e, f, a, b) = (q / 32, q / 8 % 4, q / 4 % 2, q % 4);
        t[q] = (64 * a + 16 * b + 4 * (2 * h + e) + f) as u8;
        q += 1;
    }
    t
}

const fn tile_s2(e: usize) -> [u8; 64] {
    let mut t = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        let (f, a, b) = (i / 16, i / 4 % 4, i % 4);
        t[i] = (64 * (a / 2) + 32 * (e % 2) + 8 * f + 4 * (a % 2) + b) as u8;
        i += 1;
    }
    t
}

const S1: [[u8; 64]; 2] = [tile_s1(0), tile_s1(1)];
const S2: [[u8; 64]; 4] = [tile_s2(0), tile_s2(1), tile_s2(2), tile_s2(3)];

#[inline]
#[target_feature(enable = "avx512f")]
fn cat4(a: __m128i, b: __m128i, c: __m128i, d: __m128i) -> __m512i {
    let z = _mm512_castsi128_si512(a);
    let z = _mm512_inserti32x4::<1>(z, b);
    let z = _mm512_inserti32x4::<2>(z, c);
    _mm512_inserti32x4::<3>(z, d)
}

#[inline]
#[target_feature(enable = "avx512f")]
fn lanes(y: __m512i) -> [__m128i; 4] {
    [
        _mm512_extracti32x4_epi32::<0>(y),
        _mm512_extracti32x4_epi32::<1>(y),
        _mm512_extracti32x4_epi32::<2>(y),
        _mm512_extracti32x4_epi32::<3>(y),
    ]
}

fn tile(dst: &mut [u8], ds: usize, src: &[u8], ss: usize, clockwise: bool) -> bool {
    // SAFETY: only reachable through `detect`, which checked the AVX2, AVX-512 F, BW and VBMI
    // features before this fn pointer could be handed out.
    let done = unsafe {
        if clockwise {
            tile_kernel::<true>(dst, ds, src, ss)
        } else {
            tile_kernel::<false>(dst, ds, src, ss)
        }
    };
    done.is_some()
}

/// Transposes the 16x16 block at the start of `src` into 16 rows of `dst`. Clockwise reads the
/// block's rows bottom to top (dst row t is column t, read upwards); counter-clockwise reads them
/// top to bottom and writes the columns in reverse (dst row t is column 15-t). None if a row of
/// either is out of the slice.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
fn tile_kernel<const CW: bool>(dst: &mut [u8], ds: usize, src: &[u8], ss: usize) -> Option<()> {
    let mut x = [_mm_setzero_si128(); 16];
    for (r, v) in x.iter_mut().enumerate() {
        let row = if CW { 15 - r } else { r };
        *v = ld128(at::<16>(src, row * ss)?);
    }
    let z = [
        cat4(x[0], x[1], x[2], x[3]),
        cat4(x[4], x[5], x[6], x[7]),
        cat4(x[8], x[9], x[10], x[11]),
        cat4(x[12], x[13], x[14], x[15]),
    ];
    let (s1h0, s1h1) = (ld512(&S1[0]), ld512(&S1[1]));
    // v[2p + h] is V_{p,h}.
    let v = [
        _mm512_permutex2var_epi8(z[0], s1h0, z[1]),
        _mm512_permutex2var_epi8(z[0], s1h1, z[1]),
        _mm512_permutex2var_epi8(z[2], s1h0, z[3]),
        _mm512_permutex2var_epi8(z[2], s1h1, z[3]),
    ];
    let y = [
        _mm512_permutex2var_epi8(v[0], ld512(&S2[0]), v[2]),
        _mm512_permutex2var_epi8(v[0], ld512(&S2[1]), v[2]),
        _mm512_permutex2var_epi8(v[1], ld512(&S2[2]), v[3]),
        _mm512_permutex2var_epi8(v[1], ld512(&S2[3]), v[3]),
    ];
    let cols = [lanes(y[0]), lanes(y[1]), lanes(y[2]), lanes(y[3])];
    for (e, l) in cols.iter().enumerate() {
        for (f, &v) in l.iter().enumerate() {
            let col = 4 * e + f;
            let t = if CW { col } else { 15 - col };
            st128(at_mut::<16>(dst, t * ds)?, v);
        }
    }
    Some(())
}

// ---- resample, vertical pass ----------------------------------------------------------------

fn vert(out: &mut [u8], mid: &[u8], mrow: usize, first: usize, ws: &[i16]) -> usize {
    // SAFETY: only reachable through `detect`, which checked the AVX2, AVX-512 F, BW and VBMI
    // features before this fn pointer could be handed out.
    unsafe { vert_kernel(out, mid, mrow, first, ws) }
}

/// 64 columns per step, the taps two rows at a time: the bytes of two rows are interleaved and
/// widened to (a, b) 16-bit pairs per 32-bit lane, VPMADDWD with the pair (w0, w1) gives
/// a*w0 + b*w1 exactly (at most 2 * 255 * 32768), and the sums are added to 8192 (ONE/2) in i32,
/// as the scalar code. The shift by 14 is arithmetic; VPACKSSDW then VPACKUSWB saturate, which
/// equals clamp(0, 255) of the shifted value. An odd last tap pairs with a zero row.
#[target_feature(enable = "avx512f,avx512bw")]
fn vert_kernel(out: &mut [u8], mid: &[u8], mrow: usize, first: usize, ws: &[i16]) -> usize {
    let zero = _mm512_setzero_si512();
    let init = _mm512_set1_epi32(8192);
    let mut done = 0;
    for d in out.as_chunks_mut::<64>().0 {
        let mut acc = [init; 4];
        for (k, w) in ws.chunks(2).enumerate() {
            let (w0, w1) = match *w {
                [a, b] => (a, b),
                [a] => (a, 0),
                _ => (0, 0),
            };
            let wv = _mm512_set1_epi32((i32::from(w1) << 16) | i32::from(w0.cast_unsigned()));
            let row = first + 2 * k;
            let Some(p) = at::<64>(mid, row * mrow + done) else {
                return done;
            };
            let a = ld512(p);
            let b = if w.len() == 2 {
                let Some(q) = at::<64>(mid, (row + 1) * mrow + done) else {
                    return done;
                };
                ld512(q)
            } else {
                zero
            };
            let lo = _mm512_unpacklo_epi8(a, b);
            let hi = _mm512_unpackhi_epi8(a, b);
            let parts = [
                _mm512_unpacklo_epi8(lo, zero),
                _mm512_unpackhi_epi8(lo, zero),
                _mm512_unpacklo_epi8(hi, zero),
                _mm512_unpackhi_epi8(hi, zero),
            ];
            for (s, p) in acc.iter_mut().zip(parts) {
                *s = _mm512_add_epi32(*s, _mm512_madd_epi16(p, wv));
            }
        }
        let s0 = _mm512_srai_epi32::<14>(acc[0]);
        let s1 = _mm512_srai_epi32::<14>(acc[1]);
        let s2 = _mm512_srai_epi32::<14>(acc[2]);
        let s3 = _mm512_srai_epi32::<14>(acc[3]);
        st512(
            d,
            _mm512_packus_epi16(_mm512_packs_epi32(s0, s1), _mm512_packs_epi32(s2, s3)),
        );
        done += 64;
    }
    done
}
