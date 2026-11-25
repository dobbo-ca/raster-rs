# Dependencies

- `rayon` `=1.12.0`: the worker pool that the caller's `ThreadPool` and the crate's lazy shared
  pool run on. Without it each call would start its own threads, which costs tens of
  microseconds each and makes small images slower than Go. The version is pinned exactly. It
  brings `rayon-core`, `crossbeam-deque`, `crossbeam-epoch`, `crossbeam-utils` and `either`;
  nothing else is a dependency.
