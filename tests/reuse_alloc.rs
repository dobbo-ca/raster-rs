//! Buffer reuse: with `max_threads` 1, a second call of the same size makes no allocator call, for
//! every op; a reused `Resampler` gives the bytes and errors of the free `resample` for any
//! history of shapes, and fails `AllocLimit` where the free function does.
mod common;
use common::*;
use raster::{Error, Limits, Resampler, luma, resample, rotate90, scale2to1};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;

// Allocator calls made by the current thread. A call at `max_threads` 1 runs on the test's own
// thread, so other threads (the test harness) do not count. A const-initialised `Cell<u64>` has
// no destructor and no lazy setup, so touching it inside the allocator allocates nothing.
thread_local! {
    static CALLS: Cell<u64> = const { Cell::new(0) };
}
// Tests of this file run one at a time, as in tests/alloc.rs.
static SERIAL: Mutex<()> = Mutex::new(());

struct Counting;

fn count() {
    let _ = CALLS.try_with(|c| c.set(c.get() + 1));
}

// SAFETY: forwards every call to `System` unchanged; the counter is a thread-local `Cell` that
// neither allocates nor panics.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        count();
        // SAFETY: the caller's contract for `alloc` is passed through.
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        count();
        // SAFETY: the caller's contract for `alloc_zeroed` is passed through.
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` and `l` come from the caller's matching `alloc`.
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        count();
        // SAFETY: the caller's contract for `realloc` is passed through.
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static A: Counting = Counting;

fn lim(max_alloc: usize, max_threads: usize) -> Limits {
    Limits {
        max_alloc,
        max_pixels: u64::MAX,
        max_threads,
        ..Limits::default()
    }
}

/// Runs `f` once to warm up (lazy statics, buffers), then again; the second run must succeed and
/// make no allocator call. With `warm_allocs`, the warm-up must have made some (a positive control
/// that the counter counts: a fresh `Resampler` has to allocate its buffers).
fn no_alloc_second_call(name: &str, warm_allocs: bool, mut f: impl FnMut() -> Result<(), Error>) {
    let start = CALLS.with(Cell::get);
    f().unwrap_or_else(|e| panic!("{name}: warm-up {e:?}"));
    if warm_allocs {
        assert!(
            CALLS.with(Cell::get) > start,
            "{name}: the warm-up made no allocator call, so the counter does not count"
        );
    }
    let before = CALLS.with(Cell::get);
    let r = f();
    let calls = CALLS.with(Cell::get) - before;
    assert_eq!(r, Ok(()), "{name}");
    assert_eq!(
        calls, 0,
        "{name}: {calls} allocator calls in the second call"
    );
}

#[test]
fn second_call_allocates_nothing() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let l = lim(1 << 28, 1);
    for ch in [1, 3] {
        let src = plane(1, 200, 150, ch, 3);
        let mut dst = blank(100, 75, ch, 2);
        no_alloc_second_call(&format!("scale2to1 ch {ch}"), false, || {
            scale2to1(&mut dst.plane_mut(), &src.plane(), &l, &NO_CANCEL)
        });
        for cw in [true, false] {
            let mut dst = blank(150, 200, ch, 1);
            no_alloc_second_call(&format!("rotate90 ch {ch} cw {cw}"), false, || {
                rotate90(&mut dst.plane_mut(), &src.plane(), cw, &l, &NO_CANCEL)
            });
        }
        // Down and up, on one Resampler for each size pair.
        for (dw, dh) in [(90, 70), (310, 240)] {
            let mut rs = Resampler::default();
            let mut dst = blank(dw, dh, ch, 2);
            no_alloc_second_call(
                &format!("Resampler 200x150 to {dw}x{dh} ch {ch}"),
                true,
                || rs.resample(&mut dst.plane_mut(), &src.plane(), &l, &NO_CANCEL),
            );
        }
    }
    let rgb = plane(2, 200, 150, 3, 0);
    let mut grey = blank(200, 150, 1, 4);
    no_alloc_second_call("luma", false, || {
        luma(&mut grey.plane_mut(), &rgb.plane(), &l, &NO_CANCEL)
    });
}

/// A shape that shrinks, grows, repeats and changes one axis only gives the free function's bytes
/// at every step, at 1 and 4 threads, and after a cancelled call.
#[test]
fn reused_output_equals_free_function() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    const SEQ: [(usize, usize, usize, usize); 9] = [
        (300, 200, 150, 100),
        (120, 90, 60, 45),
        (120, 90, 60, 45),
        (1200, 900, 400, 300),
        (400, 300, 400, 300),
        (300, 200, 150, 100),
        (300, 260, 150, 120),
        (80, 60, 160, 120),
        (301, 199, 77, 151),
    ];
    for threads in [1, 4] {
        let l = lim(1 << 28, threads);
        for ch in [1, 3] {
            let mut rs = Resampler::default();
            for (i, (sw, sh, dw, dh)) in SEQ.into_iter().enumerate() {
                let src = plane(i as u64 + 1, sw, sh, ch, 3);
                let mut want = blank(dw, dh, ch, 2);
                let mut got = blank(dw, dh, ch, 2);
                resample(&mut want.plane_mut(), &src.plane(), &l, &NO_CANCEL).unwrap();
                rs.resample(&mut got.plane_mut(), &src.plane(), &l, &NO_CANCEL)
                    .unwrap();
                assert!(
                    got.pix == want.pix,
                    "threads {threads} ch {ch} step {i}: bytes differ"
                );
            }
            // A cancelled call fails alike and leaves the Resampler usable.
            let src = plane(9, 300, 200, ch, 0);
            let stop = AtomicBool::new(true);
            let mut a = blank(150, 100, ch, 0);
            let mut b = blank(150, 100, ch, 0);
            assert_eq!(
                resample(&mut a.plane_mut(), &src.plane(), &l, &stop),
                Err(Error::Cancelled)
            );
            assert_eq!(
                rs.resample(&mut b.plane_mut(), &src.plane(), &l, &stop),
                Err(Error::Cancelled)
            );
            resample(&mut a.plane_mut(), &src.plane(), &l, &NO_CANCEL).unwrap();
            rs.resample(&mut b.plane_mut(), &src.plane(), &l, &NO_CANCEL)
                .unwrap();
            assert!(a.pix == b.pix, "threads {threads} ch {ch}: after cancel");
        }
    }
}

/// Kept capacity never turns a passing call into a failing one: a Resampler that holds the buffers
/// of a big call still passes a small call at exactly its need, and fails one byte below it with
/// the free function's error; after that failure it works again.
#[test]
fn reused_resampler_fails_alloc_limit_like_free_function() {
    let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for threads in [1, 4] {
        for ch in [1, 3] {
            let big = plane(1, 1200, 900, ch, 0);
            let small = plane(2, 300, 200, ch, 0);
            let mut dst_big = blank(600, 450, ch, 0);
            let mut want = blank(150, 100, ch, 0);
            let Err(Error::AllocLimit { got: need, .. }) = resample(
                &mut want.plane_mut(),
                &small.plane(),
                &lim(0, threads),
                &NO_CANCEL,
            ) else {
                panic!("a limit of 0 must refuse")
            };
            let need = need as usize;
            let mut rs = Resampler::default();
            rs.resample(
                &mut dst_big.plane_mut(),
                &big.plane(),
                &lim(1 << 28, threads),
                &NO_CANCEL,
            )
            .unwrap();
            // The same Resampler, now with a limit only the small call fits in.
            let mut got = blank(150, 100, ch, 0);
            let at = format!("threads {threads} ch {ch}");
            resample(
                &mut want.plane_mut(),
                &small.plane(),
                &lim(need, threads),
                &NO_CANCEL,
            )
            .unwrap();
            rs.resample(
                &mut got.plane_mut(),
                &small.plane(),
                &lim(need, threads),
                &NO_CANCEL,
            )
            .unwrap_or_else(|e| panic!("{at}: {e:?}"));
            assert!(got.pix == want.pix, "{at}: bytes differ");
            let tight = lim(need - 1, threads);
            let free = resample(&mut want.plane_mut(), &small.plane(), &tight, &NO_CANCEL);
            let reused = rs.resample(&mut got.plane_mut(), &small.plane(), &tight, &NO_CANCEL);
            assert_eq!(
                free,
                Err(Error::AllocLimit {
                    got: need as u64,
                    max: need as u64 - 1
                }),
                "{at}"
            );
            assert_eq!(reused, free, "{at}");
            rs.resample(
                &mut got.plane_mut(),
                &small.plane(),
                &lim(need, threads),
                &NO_CANCEL,
            )
            .unwrap_or_else(|e| panic!("{at}: after failure {e:?}"));
            assert!(got.pix == want.pix, "{at}: bytes differ after failure");
        }
    }
}
