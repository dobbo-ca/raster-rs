//! Ports of raster-go's TestResampleScratchExact and TestResampleScratchLimit: an allowed resample
//! allocates no more than the need it reported, and a refused one allocates (almost) nothing.
mod common;
use common::*;
use raster::{Error, Limits, resample};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

// Live bytes and their peak, over all threads: worker allocations count too.
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
// The counters are global, so the tests of this file run one at a time.
static SERIAL: Mutex<()> = Mutex::new(());

struct Counting;

fn grow(n: u64) {
    let live = LIVE.fetch_add(n, Relaxed) + n;
    PEAK.fetch_max(live, Relaxed);
}

// SAFETY: forwards every call to `System` unchanged; the counters are atomics, which neither
// allocate nor panic.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        grow(l.size() as u64);
        // SAFETY: the caller's contract for `alloc` is passed through.
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        grow(l.size() as u64);
        // SAFETY: the caller's contract for `alloc_zeroed` is passed through.
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size() as u64, Relaxed);
        // SAFETY: `p` and `l` come from the caller's matching `alloc`.
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        LIVE.fetch_sub(l.size() as u64, Relaxed);
        grow(n as u64);
        // SAFETY: the caller's contract for `realloc` is passed through.
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static A: Counting = Counting;

const SLACK: u64 = 1 << 16;
const SHAPES: [(usize, usize, usize, usize); 3] = [
    (200_000, 4, 1, 1),
    (4, 200_000, 1, 1),
    (3000, 2000, 700, 500),
];

/// The most bytes live at once above the level at the start, while `f` runs, on any thread.
fn peak_live<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let base = LIVE.load(Relaxed);
    PEAK.store(base, Relaxed);
    let r = f();
    (r, PEAK.load(Relaxed) - base)
}

fn lim(max_alloc: u64, max_threads: usize) -> Limits {
    Limits {
        max_alloc: max_alloc as usize,
        max_pixels: u64::MAX,
        max_threads,
        ..Limits::default()
    }
}

#[test]
fn resample_allocates_within_reported_need() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // Start the shared pool and the workers' own state before measuring.
    let warm = plane(1, 1000, 800, 1, 0);
    let mut warm_dst = blank(500, 400, 1, 0);
    resample(
        &mut warm_dst.plane_mut(),
        &warm.plane(),
        &lim(1 << 28, 4),
        &NO_CANCEL,
    )
    .unwrap();
    for threads in [1, 4] {
        for (sw, sh, dw, dh) in SHAPES {
            let src = plane(1, sw, sh, 1, 0);
            let mut dst = blank(dw, dh, 1, 0);
            let (probe, refused) = peak_live(|| {
                resample(
                    &mut dst.plane_mut(),
                    &src.plane(),
                    &lim(0, threads),
                    &NO_CANCEL,
                )
            });
            let Err(Error::AllocLimit { got, .. }) = probe else {
                panic!("{probe:?}")
            };
            assert!(refused < SLACK, "refused call allocated {refused}");
            let (r, used) = peak_live(|| {
                resample(
                    &mut dst.plane_mut(),
                    &src.plane(),
                    &lim(got, threads),
                    &NO_CANCEL,
                )
            });
            r.unwrap();
            assert!(
                used <= got + SLACK,
                "{sw}x{sh} threads {threads}: peak {used}, need {got}"
            );
        }
    }
}

/// A wide RGB shape, whose accumulator row (360 KB) is far above SLACK: the peak must stay within
/// `max_alloc` as the number of bands, and so of accumulator rows, grows.
#[test]
fn resample_wide_rows_stay_within_max_alloc() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let (sw, sh, dw, dh) = (40_000, 40, 30_000, 40);
    let acc = (dw * 3 * 4) as u64;
    let src = plane(1, sw, sh, 3, 0);
    let mut dst = blank(dw, dh, 3, 0);
    let probe = resample(&mut dst.plane_mut(), &src.plane(), &lim(0, 1), &NO_CANCEL);
    let Err(Error::AllocLimit { got: need, .. }) = probe else {
        panic!("{probe:?}")
    };
    // Start the shared pool and the workers' own state before measuring.
    resample(
        &mut dst.plane_mut(),
        &src.plane(),
        &lim(need + 3 * acc, 8),
        &NO_CANCEL,
    )
    .unwrap();
    let want = dst.pix.clone();
    for threads in [1, 2, 4, 8] {
        for max_alloc in [need, need + acc, need + 3 * acc] {
            dst.pix.fill(0);
            let (r, used) = peak_live(|| {
                resample(
                    &mut dst.plane_mut(),
                    &src.plane(),
                    &lim(max_alloc, threads),
                    &NO_CANCEL,
                )
            });
            r.unwrap();
            assert!(
                used <= max_alloc + SLACK,
                "threads {threads} max_alloc {max_alloc}: peak {used}"
            );
            assert!(
                dst.pix == want,
                "threads {threads} max_alloc {max_alloc}: output differs"
            );
        }
    }
}
