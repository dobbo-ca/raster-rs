//! Ports of raster-go's raster_test.go (the fuzz targets are in `tests/fuzz`).
mod common;
use common::*;
use raster::{Error, Limits, Plane, PlaneMut, luma, resample, rotate90, scale2to1};
use std::sync::atomic::{AtomicBool, Ordering};

const WIDTHS: [usize; 24] = [
    1, 2, 3, 15, 16, 17, 18, 19, 31, 32, 33, 34, 35, 63, 64, 65, 66, 67, 127, 128, 129, 130, 255,
    257,
];
// One thread: the ported Go checks run serially; the threaded paths have their own tests.
const LIM: Limits = Limits {
    max_pixels: 1 << 26,
    max_alloc: 1 << 28,
    max_output: 1 << 28,
    max_threads: 1,
    pool: None,
};

#[test]
fn default_limits_match_go() {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    assert_eq!(
        Limits::default(),
        Limits {
            max_threads: threads,
            ..LIM
        }
    );
}

#[test]
fn scale2to1_exact() {
    for ch in [1, 3] {
        for w in WIDTHS {
            for pad in [0, 3] {
                let src = plane(w as u64, w, 5, ch, pad);
                let mut dst = blank((w / 2).max(1), 2, ch, pad);
                let r = scale2to1(&mut dst.plane_mut(), &src.plane(), &LIM, &NO_CANCEL);
                if w < 2 {
                    assert!(matches!(r, Err(Error::Invalid(_))), "w={w}: {r:?}");
                    continue;
                }
                r.unwrap();
                same(&format!("ch={ch} w={w}"), &dst, |x, y, c| {
                    (src.at(2 * x, 2 * y, c)
                        + src.at(2 * x + 1, 2 * y, c)
                        + src.at(2 * x, 2 * y + 1, c)
                        + src.at(2 * x + 1, 2 * y + 1, c)
                        + 2)
                        >> 2
                });
            }
        }
    }
}

fn luma_of(r: i32, g: i32, b: i32) -> i32 {
    (38 * r + 75 * g + 15 * b + 64) >> 7
}

#[test]
fn luma_exact() {
    for w in WIDTHS {
        for pad in [0, 3] {
            let src = plane(w as u64, w, 3, 3, pad);
            let mut dst = blank(w, 3, 1, pad);
            luma(&mut dst.plane_mut(), &src.plane(), &LIM, &NO_CANCEL).unwrap();
            same(&format!("w={w}"), &dst, |x, y, _| {
                luma_of(src.at(x, y, 0), src.at(x, y, 1), src.at(x, y, 2))
            });
        }
    }
}

#[test]
fn luma_extremes() {
    for v in [0u8, 255] {
        let mut src = blank(40, 1, 3, 0);
        src.pix.fill(v);
        let mut dst = blank(40, 1, 1, 0);
        luma(&mut dst.plane_mut(), &src.plane(), &LIM, &NO_CANCEL).unwrap();
        assert!(dst.pix.iter().all(|&g| g == v), "grey of {v}");
    }
}

#[test]
fn rotate90_exact() {
    for ch in [1, 3] {
        for (w, h) in [(1, 1), (1, 7), (7, 1), (16, 16), (17, 33), (40, 5)] {
            let src = plane(9, w, h, ch, 3);
            for cw in [true, false] {
                let mut dst = blank(h, w, ch, 1);
                rotate90(&mut dst.plane_mut(), &src.plane(), cw, &LIM, &NO_CANCEL).unwrap();
                same(&format!("cw={cw} {w}x{h}"), &dst, |x, y, c| {
                    if cw {
                        src.at(y, h - 1 - x, c)
                    } else {
                        src.at(w - 1 - y, x, c)
                    }
                });
            }
        }
    }
}

fn run_resample(dst: &mut Buf, src: &Buf, lim: &Limits) -> Result<(), Error> {
    resample(&mut dst.plane_mut(), &src.plane(), lim, &NO_CANCEL)
}

#[test]
fn resample_properties() {
    for ch in [1, 3] {
        // identity is exact
        let src = plane(5, 37, 29, ch, 2);
        let mut dst = blank(37, 29, ch, 0);
        run_resample(&mut dst, &src, &LIM).unwrap();
        same("identity", &dst, |x, y, c| src.at(x, y, c));
        // a flat plane stays flat at any ratio, up or down
        for v in [0u8, 77, 255] {
            let mut flat = blank(64, 48, ch, 0);
            flat.pix.fill(v);
            for (dw, dh) in [(32, 24), (5, 3), (1, 1), (100, 90), (64, 1)] {
                let mut out = blank(dw, dh, ch, 0);
                run_resample(&mut out, &flat, &LIM).unwrap();
                same(&format!("flat {v} -> {dw}x{dh}"), &out, |_, _, _| {
                    i32::from(v)
                });
            }
        }
    }
    // a horizontal ramp halves to a ramp, within a level
    let mut ramp = blank(256, 4, 1, 0);
    for y in 0..4 {
        for x in 0..256 {
            ramp.pix[y * 256 + x] = x as u8;
        }
    }
    let mut out = blank(128, 2, 1, 0);
    run_resample(&mut out, &ramp, &LIM).unwrap();
    for x in 4..124 {
        let want = 2 * x as i32 + 1;
        assert!(
            (out.at(x, 0, 0) - want).abs() <= 1,
            "ramp x={x}: {}, want {want}+-1",
            out.at(x, 0, 0)
        );
    }
}

#[test]
fn invalid_planes() {
    let ok = blank(8, 8, 1, 0);
    let half = blank(4, 4, 1, 0);
    let buf = [0u8; 64];
    let mk = |len: usize, w, h, stride, channels| (len, w, h, stride, channels);
    let bad = [
        mk(0, 0, 0, 0, 0),
        mk(64, 8, 8, 7, 1),
        mk(63, 8, 8, 8, 1),
        mk(64, 8, 8, 8, 2),
        mk(64, usize::MAX, 2, usize::MAX, 1),
        mk(64, 8, usize::MAX, 8, 1),
        mk(64, 1, 2, usize::MAX, 1),
    ];
    for (i, &(len, w, h, stride, channels)) in bad.iter().enumerate() {
        let p = Plane {
            pix: &buf[..len],
            w,
            h,
            stride,
            channels,
        };
        let mut pm = vec![0u8; len];
        let mut m = PlaneMut {
            pix: &mut pm,
            w,
            h,
            stride,
            channels,
        };
        let mut half2 = blank(4, 4, 1, 0);
        let mut ok2 = blank(8, 8, 1, 0);
        for (name, r) in [
            (
                "src",
                scale2to1(&mut half2.plane_mut(), &p, &LIM, &NO_CANCEL),
            ),
            ("dst", scale2to1(&mut m, &ok.plane(), &LIM, &NO_CANCEL)),
            ("resamp", resample(&mut m, &ok.plane(), &LIM, &NO_CANCEL)),
            (
                "resamp src",
                resample(&mut ok2.plane_mut(), &p, &LIM, &NO_CANCEL),
            ),
        ] {
            assert!(
                matches!(r, Err(Error::Invalid(_))),
                "bad plane {i} as {name}: {r:?}"
            );
        }
    }
    let mut ok2 = blank(8, 8, 1, 0);
    let r = scale2to1(&mut ok2.plane_mut(), &ok.plane(), &LIM, &NO_CANCEL);
    assert!(matches!(r, Err(Error::Invalid(_))), "wrong dst size: {r:?}");
    // shape errors of the other kernels
    let mut rgb = blank(8, 8, 3, 0);
    assert!(matches!(
        luma(&mut ok2.plane_mut(), &ok.plane(), &LIM, &NO_CANCEL),
        Err(Error::Invalid(_))
    ));
    assert!(matches!(
        resample(&mut rgb.plane_mut(), &ok.plane(), &LIM, &NO_CANCEL),
        Err(Error::Invalid(_))
    ));
    assert!(matches!(
        rotate90(&mut ok2.plane_mut(), &half.plane(), true, &LIM, &NO_CANCEL),
        Err(Error::Invalid(_))
    ));
}

/// A dimension above i32::MAX that also breaks a limit is a limit error, as in Go; `Invalid` is
/// only for what passes Go's limits (untouched zero pages, so the 2 GiB buffer costs no memory).
#[cfg(target_pointer_width = "64")]
#[test]
fn resample_huge_dimension_is_a_limit() {
    let big = vec![0u8; 1 << 31];
    let src = Plane {
        pix: &big,
        w: 1 << 31,
        h: 1,
        stride: 1 << 31,
        channels: 1,
    };
    let mut one = [0u8];
    let mut dst = PlaneMut {
        pix: &mut one,
        w: 1,
        h: 1,
        stride: 1,
        channels: 1,
    };
    let r = resample(&mut dst, &src, &LIM, &NO_CANCEL);
    assert!(matches!(r, Err(Error::PixelLimit { .. })), "{r:?}");
    let lim = Limits {
        max_pixels: u64::MAX,
        ..LIM
    };
    let r = resample(&mut dst, &src, &lim, &NO_CANCEL);
    assert!(matches!(r, Err(Error::AllocLimit { .. })), "{r:?}");
    let lim = Limits {
        max_alloc: usize::MAX,
        ..lim
    };
    let r = resample(&mut dst, &src, &lim, &NO_CANCEL);
    assert!(matches!(r, Err(Error::Invalid(_))), "{r:?}");
}

#[test]
fn limits() {
    let src = plane(1, 64, 64, 1, 0);
    let mut dst = blank(32, 32, 1, 0);
    dst.pix.fill(0xAA);
    let pix = Limits {
        max_pixels: 4095,
        ..LIM
    };
    let out = Limits {
        max_output: 1023,
        ..LIM
    };
    assert_eq!(
        scale2to1(&mut dst.plane_mut(), &src.plane(), &pix, &NO_CANCEL),
        Err(Error::PixelLimit {
            got: 4096,
            max: 4095
        })
    );
    assert_eq!(dst.pix[0], 0xAA, "pixels: dst written despite breach");
    assert_eq!(
        scale2to1(&mut dst.plane_mut(), &src.plane(), &out, &NO_CANCEL),
        Err(Error::OutputLimit {
            got: 1024,
            max: 1023
        })
    );
    assert_eq!(dst.pix[0], 0xAA, "output: dst written despite breach");
    // the exact cap passes
    let exact = Limits {
        max_pixels: 4096,
        max_output: 1024,
        ..LIM
    };
    scale2to1(&mut dst.plane_mut(), &src.plane(), &exact, &NO_CANCEL).unwrap();
    // a zero cap refuses everything
    let zero = Limits {
        max_pixels: 0,
        max_alloc: 0,
        max_output: 0,
        ..LIM
    };
    assert!(matches!(
        scale2to1(&mut dst.plane_mut(), &src.plane(), &zero, &NO_CANCEL),
        Err(Error::PixelLimit { .. })
    ));
    // a plane over the default cap with a short buffer fails validation first
    let big = Plane {
        pix: &[0],
        w: 1 << 14,
        h: (1 << 12) + 1,
        stride: 1 << 14,
        channels: 1,
    };
    let r = luma(&mut dst.plane_mut(), &big, &LIM, &NO_CANCEL);
    assert!(
        matches!(r, Err(Error::Invalid(_))),
        "short buffer must fail validation first: {r:?}"
    );
}

/// A resample over the scratch cap refuses before it allocates; a cap at the reported need passes.
#[test]
fn resample_scratch_limit() {
    let src = plane(1, 2000, 2000, 1, 0);
    let mut dst = blank(1000, 1000, 1, 0);
    let r = run_resample(
        &mut dst,
        &src,
        &Limits {
            max_alloc: 1 << 20,
            ..LIM
        },
    );
    let Err(Error::AllocLimit { got, max }) = r else {
        panic!("got {r:?}")
    };
    assert_eq!(max, 1 << 20);
    assert!(got > max);
    assert!(
        dst.pix.iter().all(|&b| b == 0),
        "dst written despite breach"
    );
    run_resample(
        &mut dst,
        &src,
        &Limits {
            max_alloc: got as usize,
            ..LIM
        },
    )
    .unwrap();
    // and one byte less is refused
    let r = run_resample(
        &mut dst,
        &src,
        &Limits {
            max_alloc: (got - 1) as usize,
            ..LIM
        },
    );
    assert!(matches!(r, Err(Error::AllocLimit { .. })));
}

/// Extreme ratios work at exactly the reported need.
#[test]
fn resample_scratch_extremes() {
    for (sw, sh, dw, dh) in [
        (200_000, 4, 1, 1),
        (4, 200_000, 1, 1),
        (3000, 2000, 700, 500),
    ] {
        let src = plane(1, sw, sh, 1, 0);
        let mut dst = blank(dw, dh, 1, 0);
        let probe = run_resample(
            &mut dst,
            &src,
            &Limits {
                max_alloc: 0,
                ..LIM
            },
        );
        let Err(Error::AllocLimit { got, .. }) = probe else {
            panic!("{probe:?}")
        };
        let lim = Limits {
            max_alloc: got as usize,
            max_pixels: u64::MAX,
            ..LIM
        };
        run_resample(&mut dst, &src, &lim).unwrap();
    }
}

#[test]
fn cancel() {
    let cancel = AtomicBool::new(true);
    let g = plane(1, 64, 64, 1, 0);
    let c = plane(1, 64, 64, 3, 0);
    let mut h = blank(32, 32, 1, 0);
    let mut full = blank(64, 64, 1, 0);
    let results = [
        (
            "scale",
            scale2to1(&mut h.plane_mut(), &g.plane(), &LIM, &cancel),
        ),
        (
            "luma",
            luma(&mut full.plane_mut(), &c.plane(), &LIM, &cancel),
        ),
        (
            "rotate",
            rotate90(&mut full.plane_mut(), &g.plane(), true, &LIM, &cancel),
        ),
        (
            "resample",
            resample(&mut h.plane_mut(), &g.plane(), &LIM, &cancel),
        ),
    ];
    for (name, r) in results {
        assert_eq!(r, Err(Error::Cancelled), "{name}");
    }
    assert!(cancel.load(Ordering::Relaxed));
}

type Case = (String, Box<dyn Fn() -> Vec<u8>>);

fn golden_cases() -> Vec<Case> {
    let mut cs: Vec<Case> = Vec::new();
    for (w, h) in [(257usize, 131usize), (640, 480)] {
        for ch in [1usize, 3] {
            cs.push((
                format!("scale2to1/{w}x{h}x{ch}"),
                Box::new(move || {
                    let mut d = blank(w / 2, h / 2, ch, 0);
                    scale2to1(
                        &mut d.plane_mut(),
                        &plane(1, w, h, ch, 0).plane(),
                        &LIM,
                        &NO_CANCEL,
                    )
                    .unwrap();
                    samples(&d)
                }),
            ));
            for (dw, dh) in [(w / 2, h / 2), (101, 77), (w + 60, h + 20), (w, 1)] {
                cs.push((
                    format!("resample/{w}x{h}x{ch}->{dw}x{dh}"),
                    Box::new(move || {
                        let mut o = blank(dw, dh, ch, 0);
                        run_resample(&mut o, &plane(2, w, h, ch, 0), &LIM).unwrap();
                        samples(&o)
                    }),
                ));
            }
            for cw in [true, false] {
                cs.push((
                    format!("rotate90/{w}x{h}x{ch}/cw={cw}"),
                    Box::new(move || {
                        let mut o = blank(h, w, ch, 0);
                        rotate90(
                            &mut o.plane_mut(),
                            &plane(3, w, h, ch, 0).plane(),
                            cw,
                            &LIM,
                            &NO_CANCEL,
                        )
                        .unwrap();
                        samples(&o)
                    }),
                ));
            }
        }
        cs.push((
            format!("luma/{w}x{h}"),
            Box::new(move || {
                let mut o = blank(w, h, 1, 0);
                luma(
                    &mut o.plane_mut(),
                    &plane(4, w, h, 3, 0).plane(),
                    &LIM,
                    &NO_CANCEL,
                )
                .unwrap();
                samples(&o)
            }),
        ));
    }
    cs
}

/// The goldens come from raster-go (generated on amd64, purego); every output must match.
#[test]
fn golden() {
    let file = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/golden/hashes.txt"
    ))
    .unwrap();
    let want: std::collections::HashMap<&str, &str> =
        file.lines().filter_map(|l| l.split_once(' ')).collect();
    let cases = golden_cases();
    assert_eq!(want.len(), cases.len(), "golden lines vs cases");
    for (name, run) in cases {
        let got = sha256_hex(&run());
        assert_eq!(
            Some(got.as_str()),
            want.get(name.as_str()).copied(),
            "{name}"
        );
    }
}

/// Port of `TestImpl`: the SIMD tier the kernels dispatch to. The CI hosts have AVX2 (x86_64) and
/// NEON (aarch64), so a detector that always says no fails here. On x86_64 the best tier is
/// AVX-512 where the host has it: `RASTER_EXPECT_AVX512=1` (set on an AVX-512 host) requires
/// it, and `RASTER_SIMD=avx2` caps it.
#[test]
fn tier_is_detected() {
    let simd = std::env::var("RASTER_SIMD").ok();
    let expect512 = std::env::var("RASTER_EXPECT_AVX512").is_ok_and(|v| v == "1");
    let got = raster::tier();
    if simd.as_deref() == Some("scalar") {
        assert_eq!(got, "generic");
    } else if cfg!(target_arch = "x86_64") {
        if simd.as_deref() == Some("avx2") {
            assert_eq!(got, "avx2");
        } else if expect512 {
            assert_eq!(got, "avx512");
        } else {
            assert!(got == "avx2" || got == "avx512", "tier {got}");
        }
    } else if cfg!(target_arch = "aarch64") {
        assert_eq!(got, "neon");
    } else {
        assert_eq!(got, "generic");
    }
}
