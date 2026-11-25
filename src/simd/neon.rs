//! NEON kernels, 16 box outputs or 16 luma pixels per step.
//!
//! The kernels are safe `target_feature` fns with no `unsafe` in them. `unsafe` is only in the
//! dispatch fns (`box_gray`, `luma`), which call a kernel once `detect` has checked NEON, and in
//! the load/store helpers, whose array-reference parameters prove the length.

use super::Kernels;
use std::arch::aarch64::*;

pub(super) fn detect() -> Option<Kernels> {
    std::arch::is_aarch64_feature_detected!("neon").then_some(Kernels {
        name: "neon",
        box_gray,
        luma,
        box_rgb: None,
        tile: None,
        vert: None,
    })
}

fn ld(s: &[u8; 16]) -> uint8x16_t {
    // SAFETY: the array type proves `s` is 16 bytes. NEON is baseline on aarch64, so calling the
    // intrinsic without `target_feature` is sound.
    unsafe { vld1q_u8(s.as_ptr()) }
}

fn ld3(s: &[u8; 48]) -> uint8x16x3_t {
    // SAFETY: the array type proves `s` is 48 bytes, what LD3 of 16 bytes per plane reads. NEON is
    // baseline on aarch64.
    unsafe { vld3q_u8(s.as_ptr()) }
}

fn st(d: &mut [u8; 16], v: uint8x16_t) {
    // SAFETY: the array type proves `d` is 16 bytes. NEON is baseline on aarch64.
    unsafe { vst1q_u8(d.as_mut_ptr(), v) }
}

// NEON is baseline on aarch64, so a newer rustc may allow the kernel call without `unsafe`.
#[allow(unused_unsafe)]
fn box_gray(dst: &mut [u8], r0: &[u8], r1: &[u8]) -> usize {
    let n = dst.len() & !15;
    // SAFETY: only reachable through `detect`, which checked NEON before this fn pointer could be
    // handed out.
    unsafe { box_kernel(&mut dst[..n], &r0[..2 * n], &r1[..2 * n]) };
    n
}

/// UADDLP/UADALP sum the 2x2 block into 16-bit lanes, RSHRN #2 is the explicit (sum+2)>>2.
#[target_feature(enable = "neon")]
fn box_kernel(dst: &mut [u8], r0: &[u8], r1: &[u8]) {
    let rows = r0
        .as_chunks::<16>()
        .0
        .as_chunks::<2>()
        .0
        .iter()
        .zip(r1.as_chunks::<16>().0.as_chunks::<2>().0);
    for (d, ([p0, p1], [q0, q1])) in dst.as_chunks_mut::<16>().0.iter_mut().zip(rows) {
        let s0 = vpadalq_u8(vpaddlq_u8(ld(p0)), ld(q0));
        let s1 = vpadalq_u8(vpaddlq_u8(ld(p1)), ld(q1));
        let v = vrshrn_high_n_u16::<2>(vrshrn_n_u16::<2>(s0), s1);
        st(d, v);
    }
}

// See `box_gray` on the `allow`.
#[allow(unused_unsafe)]
fn luma(dst: &mut [u8], src: &[u8]) -> usize {
    let n = dst.len() & !15;
    // SAFETY: only reachable through `detect`, which checked NEON before this fn pointer could be
    // handed out.
    unsafe { luma_kernel(&mut dst[..n], &src[..3 * n]) };
    n
}

/// LD3 splits 16 RGB pixels into planes; UMULL/UMLAL accumulate 38R+75G+15B in 16 bits (at most
/// 32640), RSHRN #7 adds 64 and shifts.
#[target_feature(enable = "neon")]
fn luma_kernel(dst: &mut [u8], src: &[u8]) {
    let (wr, wg, wb) = (vdupq_n_u8(38), vdupq_n_u8(75), vdupq_n_u8(15));
    for (d, px) in dst
        .as_chunks_mut::<16>()
        .0
        .iter_mut()
        .zip(src.as_chunks::<48>().0)
    {
        let t = ld3(px);
        let lo = vmlal_u8(
            vmlal_u8(
                vmull_u8(vget_low_u8(t.0), vget_low_u8(wr)),
                vget_low_u8(t.1),
                vget_low_u8(wg),
            ),
            vget_low_u8(t.2),
            vget_low_u8(wb),
        );
        let hi = vmlal_high_u8(vmlal_high_u8(vmull_high_u8(t.0, wr), t.1, wg), t.2, wb);
        let v = vrshrn_high_n_u16::<7>(vrshrn_n_u16::<7>(lo), hi);
        st(d, v);
    }
}
