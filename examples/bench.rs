//! `bench <n> [threads]` (threads 0 or missing: the default `Limits`): prints name, destination bytes and ns per call for each kernel on a 2550x3300
//! page (a letter page at 300 dpi), as `oracle -bench` does for Go. See `tests/parity/bench.sh`.

use raster::{Limits, Plane, PlaneMut};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

/// raster-go's `fixture.Page` (xorshift64* bytes with ramps and hard edges).
fn page(seed: u64, w: usize, h: usize, ch: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut b: Vec<u8> = (0..w * h * ch)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 56) as u8
        })
        .collect();
    for y in 0..h {
        for x in 0..w {
            for c in 0..ch {
                let i = (y * w + x) * ch + c;
                b[i] = match (x / 16 + y / 16) % 5 {
                    0 => 255,
                    1 => (x * 3 + y * 5 + c * 40) as u8,
                    2 => 0,
                    _ => b[i] & 0x1f | (x + y) as u8,
                };
            }
        }
    }
    b
}

type Kernel = fn(&mut PlaneMut, &Plane, &Limits, &AtomicBool) -> Result<(), raster::Error>;

fn rotate_cw(d: &mut PlaneMut, s: &Plane, l: &Limits, c: &AtomicBool) -> Result<(), raster::Error> {
    raster::rotate90(d, s, true, l, c)
}

fn main() {
    let n: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1);
    let threads: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let (w, h) = (2550, 3300);
    let mut lim = Limits::default();
    if threads > 0 {
        lim.max_threads = threads;
    }
    let cancel = AtomicBool::new(false);
    #[allow(clippy::type_complexity)]
    let kernels: [(&str, usize, (usize, usize), usize, Kernel); 7] = [
        (
            "scale-gray",
            1,
            (w / 2, h / 2),
            1,
            raster::scale2to1 as Kernel,
        ),
        (
            "scale-rgb",
            3,
            (w / 2, h / 2),
            3,
            raster::scale2to1 as Kernel,
        ),
        ("luma", 3, (w, h), 1, raster::luma as Kernel),
        ("resample", 1, (w / 2, h / 2), 1, raster::resample as Kernel),
        (
            "resample-rgb",
            3,
            (w / 2, h / 2),
            3,
            raster::resample as Kernel,
        ),
        ("rotate", 1, (h, w), 1, rotate_cw as Kernel),
        ("rotate-rgb", 3, (h, w), 3, rotate_cw as Kernel),
    ];
    for (name, sch, (dw, dh), dch, f) in kernels {
        let pix = page(1, w, h, sch);
        let src = Plane {
            pix: &pix,
            w,
            h,
            stride: w * sch,
            channels: sch,
        };
        let mut out = vec![0u8; dw * dh * dch];
        let mut dst = PlaneMut {
            pix: &mut out,
            w: dw,
            h: dh,
            stride: dw * dch,
            channels: dch,
        };
        let run = |dst: &mut PlaneMut| {
            f(dst, &src, &lim, &cancel).expect("bench call");
            std::hint::black_box(&dst.pix);
        };
        run(&mut dst);
        let t = Instant::now();
        for _ in 0..n {
            run(&mut dst);
        }
        println!(
            "{name}\t{}\t{}",
            dw * dh * dch,
            t.elapsed().as_nanos() / u128::from(n)
        );
    }
}
