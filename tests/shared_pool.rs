//! The crate's shared pool is lazy: it is built only by a call with `max_threads` above 1, no pool
//! of its own and work for a second thread. This is its own test binary with one test, so no
//! other test has built the pool first.

use raster::{Limits, Plane, PlaneMut, ThreadPool, luma, shared_pool_started};
use std::sync::atomic::AtomicBool;

#[test]
fn shared_pool_is_lazy() {
    let cancel = AtomicBool::new(false);
    // Big enough for two bands on the SIMD path too (16 pixels weigh one unit there).
    let (w, h) = (2048, 2048);
    let src = vec![7u8; w * h * 3];
    let s = Plane {
        pix: &src,
        w,
        h,
        stride: w * 3,
        channels: 3,
    };
    let big = |l: &Limits| {
        let mut pix = vec![0u8; w * h];
        let mut d = PlaneMut {
            pix: &mut pix,
            w,
            h,
            stride: w,
            channels: 1,
        };
        luma(&mut d, &s, l, &cancel).unwrap();
    };
    let threads = Limits {
        max_threads: 4,
        ..Limits::default()
    };
    // A tiny image has no work for a second thread.
    let tiny_src = [9u8; 12];
    let ts = Plane {
        pix: &tiny_src,
        w: 2,
        h: 2,
        stride: 6,
        channels: 3,
    };
    let mut tiny = [0u8; 4];
    let mut td = PlaneMut {
        pix: &mut tiny,
        w: 2,
        h: 2,
        stride: 2,
        channels: 1,
    };
    luma(&mut td, &ts, &threads, &cancel).unwrap();
    assert!(!shared_pool_started(), "tiny image");
    big(&Limits {
        max_threads: 1,
        ..Limits::default()
    });
    assert!(!shared_pool_started(), "max_threads 1");
    big(&Limits {
        pool: Some(ThreadPool::new(2).unwrap()),
        ..threads.clone()
    });
    assert!(!shared_pool_started(), "caller pool");
    big(&threads);
    assert!(shared_pool_started(), "max_threads 4, no pool");
}
