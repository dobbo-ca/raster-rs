use crate::Error;
use crate::parallel::{ThreadPool, available_parallelism, pool_for};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Caps on one call. Defaults match raster-go where Go has the cap; unlike Go, zero is a real cap
/// (nothing passes), not "use the default". There is no package budget: the caller reserves
/// `max_alloc` from its own.
///
/// ```
/// use raster::Limits;
/// // A caller in a 2-CPU pod with a memory limit of 1 GiB.
/// let limits = Limits { max_threads: 2, max_alloc: 768 << 20, ..Limits::default() };
/// assert_eq!(limits.max_pixels, 1 << 26);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Limits {
    /// Pixels per call: the larger of source and destination.
    pub max_pixels: u64,
    /// The peak working memory in bytes of one call, all threads together (only `resample`
    /// allocates; a [`Resampler`](crate::Resampler) also counts the buffers it keeps, at their
    /// capacity): the tap tables, the weight row, the intermediate plane and one i32 accumulator
    /// row for each band of the vertical pass running at once. Each is charged here before it is
    /// allocated and released when it is freed. Go's up-front check stays: a call whose
    /// single-thread need (as `resample` reports it in
    /// [`Error::AllocLimit`](crate::Error::AllocLimit)) is over it is refused before anything is
    /// allocated. Beyond that need, extra bands are only started while their accumulator rows
    /// fit, so a call that fits at 1 thread succeeds at every `max_threads`, with the same
    /// output. Default `1 << 28`.
    pub max_alloc: usize,
    /// Bytes written to the destination.
    pub max_output: u64,
    /// The most OS threads one call uses, the calling thread included. Default
    /// `std::thread::available_parallelism()`, or 1 if that is unknown. 1 (and 0, which is treated
    /// as 1) runs everything on the calling thread: no thread is spawned and no pool is touched.
    /// There is no error for it: the call uses fewer threads. The output of an `Ok` call does
    /// not depend on it. After [`Error::Cancelled`](crate::Error::Cancelled) only rows finished
    /// before the flag was seen are written, and which rows depends on `max_threads`.
    pub max_threads: usize,
    /// The worker threads the call runs its parallel stages on, taking at most `max_threads - 1`
    /// of them besides the calling thread. Default `None`: a call with `max_threads` above 1 then
    /// uses the crate's shared pool, built on the first such call that has work for a second
    /// thread, with `available_parallelism` threads and never more. A pool given here always wins
    /// and the shared one is not built for the call. Create one with [`ThreadPool::new`] and keep
    /// it. There is no error for it.
    pub pool: Option<ThreadPool>,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_pixels: 1 << 26,
            max_alloc: 1 << 28,
            max_output: 1 << 28,
            max_threads: available_parallelism(),
            pool: None,
        }
    }
}

/// A band of a parallel stage has at least this much work. A unit is about 1 ns of work: one pixel
/// of the scalar kernels, rotate without a tile kernel and resample; the SIMD luma and scale2to1 rows (grey on every tier, RGB
/// on AVX-512) and the grey rotate90 tile path (AVX-512) weigh a row with [`simd_row_work`]. 32K units is about 33 us, where handing a task to a running pool worker
/// costs a few us. Under `--cfg fuzzing` it is 64, so the small images a fuzz input can afford
/// reach the threaded paths.
const MIN_BAND_WORK: usize = if cfg!(fuzzing) { 1 << 6 } else { 1 << 15 };

/// The work units of an `n`-pixel row on a SIMD path: the whole lanes cost about 1/16 ns per
/// pixel, the scalar tail (under 32 samples) about 1 ns. Under `--cfg fuzzing` every pixel is a
/// unit, so fuzz inputs split SIMD rows into bands too.
pub(crate) fn simd_row_work(n: usize) -> usize {
    if cfg!(fuzzing) { n } else { n / 16 + n % 32 }
}

/// What every kernel gets: the call's ledger and its cancel flag.
#[derive(Clone, Copy)]
pub(crate) struct Ctx<'a> {
    pub bud: &'a Budget,
    pub cancel: &'a AtomicBool,
}

/// The memory and threads one call has taken: a ledger of the bytes held against `max_alloc` (and
/// their peak) and of the threads live against `max_threads`. It is shared by reference across the
/// threads of the call.
pub(crate) struct Budget {
    max: u64,
    held: AtomicU64,
    #[cfg_attr(not(test), allow(dead_code))]
    peak: AtomicU64,
    max_threads: usize,
    /// Live threads, the caller's included.
    threads: AtomicUsize,
    #[cfg_attr(not(test), allow(dead_code))]
    peak_threads: AtomicUsize,
    /// The caller's pool, if it gave one.
    caller_pool: Option<ThreadPool>,
    /// Where `par_rows` hands work, resolved at the first stage that has work for a second
    /// thread, so a small image never builds the shared pool. `None` runs everything on the
    /// calling thread.
    pool: OnceLock<Option<ThreadPool>>,
}

/// Bytes taken with [`Budget::charge`], released on drop.
pub(crate) struct Charge<'a> {
    b: &'a Budget,
    n: u64,
}

impl Drop for Charge<'_> {
    fn drop(&mut self) {
        self.b.held.fetch_sub(self.n, Ordering::Relaxed);
    }
}

/// A thread slot taken with `Budget::take_slot`, given back on drop.
struct Slot<'a>(&'a Budget);

impl Drop for Slot<'_> {
    fn drop(&mut self) {
        self.0.threads.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Budget {
    pub(crate) fn new(l: &Limits) -> Budget {
        // The cancel-check hook counts checks on the calling thread, so an armed call runs
        // serially.
        #[cfg(any(test, feature = "cancel-hook"))]
        let max_threads = if crate::CANCEL_AT.with(std::cell::Cell::get) != 0 {
            1
        } else {
            l.max_threads.max(1)
        };
        #[cfg(not(any(test, feature = "cancel-hook")))]
        let max_threads = l.max_threads.max(1);
        Budget {
            max: l.max_alloc as u64,
            held: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            max_threads,
            threads: AtomicUsize::new(1),
            peak_threads: AtomicUsize::new(1),
            caller_pool: l.pool.clone(),
            pool: OnceLock::new(),
        }
    }

    /// Charges `n` bytes until the returned guard is dropped, before they are allocated.
    pub(crate) fn charge(&self, n: u64) -> Result<Charge<'_>, Error> {
        let mut cur = self.held.load(Ordering::Relaxed);
        loop {
            let got = cur.saturating_add(n);
            if got > self.max {
                return Err(Error::AllocLimit { got, max: self.max });
            }
            match self
                .held
                .compare_exchange_weak(cur, got, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => {
                    self.peak.fetch_max(got, Ordering::Relaxed);
                    return Ok(Charge { b: self, n });
                }
                Err(c) => cur = c,
            }
        }
    }

    /// The most bytes held at once so far.
    #[cfg(test)]
    pub(crate) fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }

    /// The bytes held now.
    #[cfg(test)]
    pub(crate) fn held(&self) -> u64 {
        self.held.load(Ordering::Relaxed)
    }

    /// The most threads live at once so far, the caller's included.
    #[cfg(test)]
    pub(crate) fn peak_threads(&self) -> usize {
        self.peak_threads.load(Ordering::Relaxed)
    }

    /// The threads live now, the caller's included.
    #[cfg(test)]
    pub(crate) fn threads(&self) -> usize {
        self.threads.load(Ordering::Relaxed)
    }

    /// The pool this call hands work to: the caller's, else (with `max_threads` above 1) the
    /// shared one, built now if this is the first call in the process that needs it.
    pub(crate) fn pool(&self) -> Option<&ThreadPool> {
        if self.max_threads <= 1 {
            return None;
        }
        self.pool
            .get_or_init(|| pool_for(&self.caller_pool, self.max_threads))
            .as_ref()
    }

    /// Takes one more thread if `max_threads` allows.
    fn take_slot(&self) -> Option<Slot<'_>> {
        let mut cur = self.threads.load(Ordering::Relaxed);
        loop {
            if cur >= self.max_threads {
                return None;
            }
            match self.threads.compare_exchange_weak(
                cur,
                cur + 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.peak_threads.fetch_max(cur + 1, Ordering::Relaxed);
                    return Some(Slot(self));
                }
                Err(c) => cur = c,
            }
        }
    }

    /// How many bands `par_rows_with` would like for `rows` rows of `row_work` each, in units of
    /// `unit` rows: one per `MIN_BAND_WORK`, no more than there are units, and no more threads
    /// than `max_threads` and the pool give. Builds the shared pool if a second band is wanted
    /// and the caller gave none.
    pub(crate) fn bands(&self, rows: usize, row_work: usize, unit: usize) -> usize {
        let extra = (rows.saturating_mul(row_work) / MIN_BAND_WORK)
            .saturating_sub(1)
            .min(rows.div_ceil(unit.max(1)).saturating_sub(1));
        if extra == 0 {
            return 1;
        }
        match self.pool() {
            Some(pool) => 1 + extra.min(pool.threads()).min(self.max_threads - 1),
            None => 1,
        }
    }

    /// `par_rows_with` with bands of single rows and no cap.
    pub(crate) fn par_rows<T: Send>(
        &self,
        data: &mut [T],
        shape: (usize, usize, usize),
        f: impl Fn(&mut [T], usize, usize) -> Result<(), Error> + Sync,
    ) -> Result<(), Error> {
        self.par_rows_with(data, shape, (1, usize::MAX), f)
    }

    /// Calls `f(band, first_row, end_row)` over `data`, which holds `rows` rows of `row_len`
    /// elements (the last row may be shorter; `data` must end there), in bands on as many threads
    /// as `max_threads` gives. `shape` is `(row_len, rows, row_work)`: each band has at least
    /// `MIN_BAND_WORK` of work and there are no more workers than the pool has. `split` is
    /// `(unit, cap)`: every band but the last has a multiple of `unit` rows, and there are at most
    /// `cap` bands. `f` must give the same result for any split. The error is the first band's,
    /// as a serial run gives. A panic in `f` on a worker is re-raised on the caller.
    pub(crate) fn par_rows_with<T: Send>(
        &self,
        data: &mut [T],
        (row_len, rows, row_work): (usize, usize, usize),
        (unit, cap): (usize, usize),
        f: impl Fn(&mut [T], usize, usize) -> Result<(), Error> + Sync,
    ) -> Result<(), Error> {
        let unit = unit.max(1);
        let want = self.bands(rows, row_work, unit).min(cap.max(1));
        if want <= 1 {
            return f(data, 0, rows);
        }
        let Some(pool) = self.pool() else {
            return f(data, 0, rows);
        };
        let mut slots = Vec::new();
        while slots.len() + 1 < want {
            match self.take_slot() {
                Some(s) => slots.push(s),
                None => break,
            }
        }
        if slots.is_empty() {
            return f(data, 0, rows);
        }
        let band = rows.div_ceil(unit).div_ceil(slots.len() + 1) * unit;
        let mut res = vec![Ok(()); rows.div_ceil(band)];
        let f = &f;
        pool.0.in_place_scope(|s| {
            let mut slots = slots.into_iter();
            let mut chunks = data
                .chunks_mut(band * row_len)
                .zip(res.iter_mut())
                .enumerate();
            let first = chunks.next();
            for (i, (chunk, out)) in chunks {
                let (r0, r1) = (i * band, ((i + 1) * band).min(rows));
                match slots.next() {
                    Some(slot) => s.spawn(move |_| {
                        let _slot = slot;
                        *out = f(chunk, r0, r1);
                    }),
                    None => *out = f(chunk, r0, r1),
                }
            }
            if let Some((_, (chunk, out))) = first {
                *out = f(chunk, 0, band.min(rows));
            }
        });
        res.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lim(max_alloc: usize, max_threads: usize) -> Limits {
        Limits {
            max_alloc,
            max_threads,
            ..Limits::default()
        }
    }

    /// Charges add up to the limit, a charge over it fails and changes nothing, and a released
    /// charge can be taken again; the peak is the most held at once.
    #[test]
    fn ledger_charges_release_and_peak() {
        let b = Budget::new(&lim(100, 1));
        let a = b.charge(60).unwrap();
        assert_eq!(
            b.charge(41).err(),
            Some(Error::AllocLimit { got: 101, max: 100 })
        );
        let c = b.charge(40).unwrap();
        assert_eq!(b.peak(), 100);
        assert!(b.charge(1).is_err());
        drop(a);
        drop(c);
        assert_eq!(b.held(), 0);
        let d = b.charge(70).unwrap();
        drop(d);
        assert_eq!(b.peak(), 100);
        assert!(b.charge(100).is_ok());
        assert!(b.charge(101).is_err());
    }

    /// `par_rows` gives the same data for every thread count, never has more threads live than
    /// `max_threads`, and does not spawn with one.
    #[test]
    fn par_rows_within_max_threads() {
        let row = 1024;
        let rows = 4 << 10;
        let fill = |band: &mut [u32], r0: usize, r1: usize| {
            assert_eq!(band.len(), (r1 - r0) * row);
            for (i, r) in band.chunks_exact_mut(row).enumerate() {
                r.fill((r0 + i) as u32);
            }
            Ok(())
        };
        for max in [0usize, 1, 2, 3, 64] {
            let b = Budget::new(&lim(1 << 20, max));
            let mut got = vec![0u32; rows * row];
            let ids = std::sync::Mutex::new(std::collections::HashSet::new());
            b.par_rows(&mut got, (row, rows, row), |band, r0, r1| {
                ids.lock().unwrap().insert(std::thread::current().id());
                fill(band, r0, r1)
            })
            .unwrap();
            let used = ids.into_inner().unwrap().len();
            assert!(used <= max.max(1), "max_threads {max}: {used} threads");
            if max <= 1 {
                assert_eq!(used, 1, "max_threads {max}");
            }
            for (i, r) in got.chunks_exact(row).enumerate() {
                assert!(r.iter().all(|&v| v == i as u32), "max {max} row {i}");
            }
            let pool = b.pool().map_or(0, ThreadPool::threads);
            assert_eq!(
                b.peak_threads(),
                max.max(1).min(1 + pool),
                "max_threads {max}"
            );
            assert_eq!(b.threads(), 1);
        }
        // Under two bands of work is not worth a thread.
        let b = Budget::new(&lim(1 << 20, 8));
        let mut small = vec![0u32; row];
        b.par_rows(&mut small, (row, 1, MIN_BAND_WORK), |_, _, _| Ok(()))
            .unwrap();
        assert_eq!(b.peak_threads(), 1);
    }

    /// A short last row still splits (a strided image), the first band's error comes first, and
    /// the slots come back.
    #[test]
    fn short_last_row_and_errors() {
        let pool = ThreadPool::new(2).unwrap();
        let l = Limits {
            pool: Some(pool.clone()),
            ..lim(1 << 20, 3)
        };
        let b = Budget::new(&l);
        assert!(b.pool() == Some(&pool));
        let (row, rows) = (100usize, 7usize);
        let mut data = vec![0u8; (rows - 1) * row + 10];
        let r = b.par_rows(&mut data, (row, rows, MIN_BAND_WORK), |band, r0, r1| {
            for y in r0..r1 {
                band[(y - r0) * row] = y as u8 + 1;
            }
            if r0 > 0 {
                Err(Error::Cancelled)
            } else {
                Ok(())
            }
        });
        assert_eq!(r, Err(Error::Cancelled));
        assert_eq!(b.threads(), 1);
        assert_eq!(b.peak_threads(), 3);
        for y in 0..rows {
            assert_eq!(data[y * row], y as u8 + 1);
        }
    }

    /// Bands start at multiples of `unit` and are no more than `cap`.
    #[test]
    fn unit_and_cap() {
        let pool = ThreadPool::new(4).unwrap();
        let l = Limits {
            pool: Some(pool),
            ..lim(1 << 20, 8)
        };
        let (row, rows) = (4usize, 100usize);
        for cap in [1usize, 2, 3, 99] {
            let b = Budget::new(&l);
            let starts = std::sync::Mutex::new(Vec::new());
            let mut data = vec![0u8; rows * row];
            b.par_rows_with(
                &mut data,
                (row, rows, MIN_BAND_WORK),
                (16, cap),
                |_, r0, _| {
                    starts.lock().unwrap().push(r0);
                    Ok(())
                },
            )
            .unwrap();
            let starts = starts.into_inner().unwrap();
            assert!(starts.len() <= cap, "cap {cap}: {starts:?}");
            assert!(starts.iter().all(|s| s % 16 == 0), "cap {cap}: {starts:?}");
        }
    }
}
