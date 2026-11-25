//! The worker threads of a call: a rayon pool (the caller's, in `Limits::pool`, or the crate's
//! lazy shared one) that `Budget::par_rows` hands work to. The workers are started once, not per
//! call and not per stage, so a call pays microseconds to hand a task over.

use crate::Error;
use std::sync::{Arc, OnceLock};

/// Worker threads that calls share. Create one, keep it, and pass it in
/// [`Limits::pool`](crate::Limits::pool): each call then runs its parallel stages on these
/// threads instead of the crate's shared pool. A call without one and with `max_threads` above 1
/// uses the shared pool, built on first use with `available_parallelism` threads and never more;
/// a caller's pool always wins. `Limits::max_threads` still caps the threads one call uses, the
/// calling thread included. Clones share the threads, which stop when the last clone drops.
#[derive(Clone, Debug)]
pub struct ThreadPool(pub(crate) Arc<rayon::ThreadPool>);

impl ThreadPool {
    /// A pool of `threads` workers (at least 1), started now.
    ///
    /// # Errors
    /// [`Error::OutOfMemory`] if the operating system will not start the threads. No limit of
    /// [`Limits`](crate::Limits) applies: the pool is made before any call.
    pub fn new(threads: usize) -> Result<ThreadPool, Error> {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .thread_name(|i| format!("raster-worker-{i}"))
            .build()
            .map(|p| ThreadPool(Arc::new(p)))
            .map_err(|_| Error::OutOfMemory)
    }

    /// How many worker threads the pool has.
    ///
    /// # Errors
    ///
    /// None: it cannot fail, and no limit of [`Limits`](crate::Limits) applies.
    pub fn threads(&self) -> usize {
        self.0.current_num_threads()
    }
}

/// Two handles are equal if they are clones of the same pool.
impl PartialEq for ThreadPool {
    fn eq(&self, other: &ThreadPool) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for ThreadPool {}

/// The crate's shared pool, built on first use by a call with `max_threads` above 1 and no caller
/// pool. `None` if the build failed; such a call then runs on its own thread, serially, and so
/// does every later call in the process that has no pool of its own (the failure is kept).
static SHARED: OnceLock<Option<ThreadPool>> = OnceLock::new();

/// Whether the shared pool has been built yet (for the tests: `max_threads` 1, and a caller's
/// pool, never build it).
///
/// # Errors
///
/// None: it is a query and takes no [`Limits`](crate::Limits).
#[doc(hidden)]
pub fn shared_pool_started() -> bool {
    SHARED.get().is_some()
}

/// `std::thread::available_parallelism()`, or 1 if unknown, read once: it can query cgroups.
pub(crate) fn available_parallelism() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

/// The pool one call runs on: the caller's, else (with `max_threads` above 1) the shared one.
/// `None` runs the call on its own thread.
pub(crate) fn pool_for(pool: &Option<ThreadPool>, max_threads: usize) -> Option<ThreadPool> {
    if max_threads <= 1 {
        return None;
    }
    if let Some(p) = pool {
        return Some(p.clone());
    }
    SHARED
        .get_or_init(|| ThreadPool::new(available_parallelism()).ok())
        .clone()
}
