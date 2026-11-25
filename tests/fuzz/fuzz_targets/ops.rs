#![no_main]
use libfuzzer_sys::fuzz_target;

include!("../../parity/case.rs");

// The parity input format: hostile sizes, strides and channels, under the oracle's limits. No
// call may panic, and a call that fails other than by cancellation leaves the destination as it
// was (all zeros here); one that passes stays within the limits.
fuzz_target!(|data: &[u8]| {
    let Some(case) = Case::parse(data) else {
        return;
    };
    let limits = case.limits();
    // One thread and four give the same result and bytes (a cancel at the Nth check runs
    // serially whatever the count). The crate is built with `cfg(fuzzing)`, so small inputs split.
    let at = |max_threads| {
        case.run(&Limits {
            max_threads,
            ..limits.clone()
        })
    };
    let (r, out) = at(1);
    let (r4, out4) = at(4);
    assert_eq!(r, r4, "result differs between thread counts");
    assert!(out == out4, "bytes differ between thread counts");
    // For a resample case (op 4), one Resampler over the case and a derived one of another destination size, at 1 and 4
    // threads, in turn: each call gives the free function's result and bytes, so tap tables kept for
    // another shape, trims after a failed charge and kept accumulator rows are all compared.
    if case.op == 4 {
        let [sw, sh, sst, dw, dh, dst, dlen] = case.f;
        let other = Case {
            f: [sw, sh, sst, dw / 2 + 1, dh / 2 + 1, dst, dlen],
            ..case
        };
        let mut rs = raster::Resampler::default();
        for (c, max_threads) in [(&case, 1), (&other, 4), (&case, 4), (&other, 1), (&case, 1)] {
            let l = Limits {
                max_threads,
                ..limits.clone()
            };
            let (want, want_out) = c.run(&l);
            let (got, got_out) = c.run_with(&l, Some(&mut rs));
            assert_eq!(want, got, "reused Resampler result differs");
            assert!(want_out == got_out, "reused Resampler bytes differ");
        }
    }
    match r {
        Ok(()) | Err(Error::Cancelled) => {}
        Err(_) => assert!(out.iter().all(|&b| b == 0), "dst written on error"),
    }
});
