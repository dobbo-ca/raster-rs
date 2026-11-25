# CI image: every check runs inside this Dockerfile, so CI and a local `docker buildx build` run the same steps.
# Dependencies (rayon) are fetched in the toolchain stage, before the source is copied, and every later
# stage builds offline from that layer (so the cargo registry is not a cache mount).
# Every tool is installed in a stage before `COPY . .`, so a new commit does not install it again.
FROM rust:1.99.1-bookworm AS toolchain
RUN rustup component add rustfmt clippy && rustup target add aarch64-unknown-linux-gnu
WORKDIR /src
# The manifests and lockfiles of the crate and of the fuzz crate, with stub targets, so the fetch is
# a layer that only a change to them rebuilds.
COPY Cargo.toml Cargo.lock ./
COPY tests/fuzz/Cargo.toml tests/fuzz/Cargo.lock tests/fuzz/
RUN mkdir -p src tests/fuzz/fuzz_targets && touch src/lib.rs \
    && for t in $(sed -n 's#^path = "fuzz_targets/\(.*\)\.rs"$#\1#p' tests/fuzz/Cargo.toml); do echo 'fn main() {}' > tests/fuzz/fuzz_targets/$t.rs; done \
    && cargo fetch --locked && cargo fetch --locked --manifest-path tests/fuzz/Cargo.toml
ENV CARGO_NET_OFFLINE=true

FROM toolchain AS src
COPY . .

# The checks run in the build, so a red check fails the build; a rerun of the same source is a
# BuildKit cache hit. The target dirs are cache mounts, so they persist across builds. They are
# shared by every branch: sharing=locked lets one build at a time use a mount (a second waits), and
# the touch of the sources before every cargo RUN covers a mount that another branch wrote earlier:
# otherwise a cached artifact newer than the copied files counts as fresh and cargo builds stale code.
FROM src AS ci
RUN --mount=type=cache,id=raster-target,sharing=locked,target=/src/target \
    cargo fmt --check
RUN --mount=type=cache,id=raster-target,sharing=locked,target=/src/target \
    find src examples tests -type f -exec touch {} + && cargo clippy --locked --all-targets -- -D warnings
RUN --mount=type=cache,id=raster-target,sharing=locked,target=/src/target \
    find src examples tests -type f -exec touch {} + && cargo clippy --locked --all-targets --target aarch64-unknown-linux-gnu -- -D warnings
RUN --mount=type=cache,id=raster-target,sharing=locked,target=/src/target \
    find src examples tests -type f -exec touch {} + && cargo test --locked
RUN --mount=type=cache,id=raster-target,sharing=locked,target=/src/target \
    find src examples tests -type f -exec touch {} + && RASTER_SIMD=avx2 cargo test --locked
RUN --mount=type=cache,id=raster-target,sharing=locked,target=/src/target \
    find src examples tests -type f -exec touch {} + && RASTER_SIMD=scalar cargo test --locked
# The x86-64-v4 build (docs/performance.md, the bench's v4 column) must pass the same tests: the
# unit tests and the golden, limits and parity-vector tests of tests/raster.rs, released, with its
# own cache mount since RUSTFLAGS changes every artifact. It runs only on a host with AVX-512
# (VBMI); elsewhere it is skipped, because the binary would die with SIGILL. RASTER_EXPECT_AVX512=1
# makes the tests fail if the tier is not detected. The sources are touched first, as in every
# cargo RUN here (see above).
RUN --mount=type=cache,id=raster-target-v4,sharing=locked,target=/src/target \
    if grep -qw avx512vbmi /proc/cpuinfo; then \
      find src examples tests -type f -exec touch {} + \
      && RASTER_EXPECT_AVX512=1 RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls" cargo test --locked --release --lib --test raster; \
    else echo "no AVX-512 on this host: x86-64-v4 tests skipped"; fi
RUN --mount=type=cache,id=raster-target,sharing=locked,target=/src/target \
    find src examples tests -type f -exec touch {} + && cargo build --locked --release --target aarch64-unknown-linux-gnu

# The arm64 stage runs on a native arm64 runner (ubuntu-24.04-arm).
FROM src AS arm64
# aarch64 tests on the native host, both SIMD tiers (NEON, then RASTER_SIMD=scalar): only the tests
# that the target changes. The unit tests hold the NEON kernels against scalar and the dispatch
# tests; in tests/raster.rs, tier_is_detected checks the tier and scale2to1_exact, luma_exact and
# luma_extremes run the kernels through the API. The rest (limits, resample, rotate, golden, cancel,
# alloc) has no aarch64-specific code. The full suite runs natively in the `test` job (stage `ci`).
# The `simd::` filter picks the NEON-vs-scalar unit tests; filters apply to every test binary.
RUN --mount=type=cache,id=raster-target-arm64,sharing=locked,target=/src/target \
    find src examples tests -type f -exec touch {} + && cargo test --locked --lib --test raster -- \
    simd:: tier_is_detected scale2to1_exact luma_exact luma_extremes
RUN --mount=type=cache,id=raster-target-arm64,sharing=locked,target=/src/target \
    find src examples tests -type f -exec touch {} + && RASTER_SIMD=scalar cargo test --locked --lib --test raster -- \
    simd:: tier_is_detected scale2to1_exact luma_exact luma_extremes

# Go parity: the Go oracle writes the inputs and runs them; the Rust example runs the same inputs;
# the two output trees must be equal.
FROM golang:1.27-bookworm AS go-oracle
ENV GOTOOLCHAIN=local CGO_ENABLED=0
WORKDIR /src
COPY tests/parity/go tests/parity/go
RUN --mount=type=cache,id=raster-go-build,target=/root/.cache/go-build cd tests/parity/go/oracle && go build -o /out/oracle .
# The same oracle built for x86-64-v4 (the bench's v4 column); the parity stages use /out/oracle.
RUN --mount=type=cache,id=raster-go-build-v4,target=/root/.cache/go-build cd tests/parity/go/oracle && GOAMD64=v4 go build -o /out/oracle-v4 .

FROM go-oracle AS go-run
COPY tests/fuzz/seeds/ops tests/fuzz/seeds/ops
# Inputs: the generated ones and the ops fuzz seeds (the Go FuzzX seeds are in tests/fuzz/seeds too).
RUN /out/oracle -gen /inputs \
    && for f in tests/fuzz/seeds/ops/*; do cp $f /inputs/$(basename $f).in; done \
    && ls /inputs | wc -l && du -sh /inputs
RUN /out/oracle /inputs /go-out && ls /go-out | wc -l && du -sh /go-out

# The Go trees are bind mounts from go-run, not COPY: a COPY after `COPY . .` would copy them again
# on every commit. The Rust output is deleted in the same RUN, so the layer (and the cache export)
# stays small. Pass 1 is the default threads and detected SIMD tier (AVX-512 where the host has it), pass 2
# the scalar tier, pass 3 one thread, pass 4 the AVX2 tier; each is diffed against the same Go outputs.
FROM src AS parity
RUN --mount=type=cache,id=raster-target,sharing=locked,target=/src/target \
    --mount=type=bind,from=go-run,source=/inputs,target=/inputs \
    --mount=type=bind,from=go-run,source=/go-out,target=/go-out \
    find src examples tests -type f -exec touch {} + && cargo run --locked --release --features cancel-hook --example oracle -- /inputs /rust-out \
    && echo "$(ls /inputs | wc -l) inputs, $(ls /go-out | wc -l) outputs ($(ls /go-out | grep -c '\.out$') ok, $(ls /go-out | grep -c '\.err$') errors)" \
    && [ "$(ls /go-out | wc -l)" -gt 1000 ] \
    && { diff -rq /go-out /rust-out >/tmp/diff.txt || { head -50 /tmp/diff.txt; wc -l /tmp/diff.txt; exit 1; }; } \
    && RASTER_SIMD=scalar cargo run --locked --release --features cancel-hook --example oracle -- /inputs /rust-scalar-out \
    && { diff -rq /go-out /rust-scalar-out >/tmp/diff.txt || { head -50 /tmp/diff.txt; wc -l /tmp/diff.txt; exit 1; }; } \
    && cargo run --locked --release --features cancel-hook --example oracle -- --threads 1 /inputs /rust-1t-out \
    && { diff -rq /go-out /rust-1t-out >/tmp/diff.txt || { head -50 /tmp/diff.txt; wc -l /tmp/diff.txt; exit 1; }; } \
    && RASTER_SIMD=avx2 cargo run --locked --release --features cancel-hook --example oracle -- /inputs /rust-avx2-out \
    && { diff -rq /go-out /rust-avx2-out >/tmp/diff.txt || { head -50 /tmp/diff.txt; wc -l /tmp/diff.txt; exit 1; }; } \
    && rm -rf /rust-out /rust-scalar-out /rust-1t-out /rust-avx2-out

# Fuzzing (fuzz.yml): nightly and cargo-fuzz, with the targets built into the image. The
# workflow runs it with `docker run`, so a BuildKit cache hit cannot skip the fuzzing.
FROM toolchain AS fuzz-tools
RUN rustup toolchain install nightly --profile minimal \
    && CARGO_NET_OFFLINE=false cargo install cargo-fuzz --locked --version 0.13.1

FROM fuzz-tools AS fuzz
COPY . .
RUN cargo +nightly fuzz build --fuzz-dir tests/fuzz -a

# The fuzz-arm64 job (fuzz.yml) uses this same `fuzz` stage on a native arm64 runner, so the NEON
# kernels are fuzzed with the sanitizer.

# Benchmark (bench.yml): the Go oracle and the Rust bench example in one image, each built twice:
# the default build, and x86-64-v4 (GOAMD64=v4; RUSTFLAGS as docs/performance.md recommends) as
# oracle-v4 and bench-v4; tests/parity/bench.sh prints both.
FROM src AS bench
COPY --from=go-oracle /out/oracle /usr/local/bin/oracle
COPY --from=go-oracle /out/oracle-v4 /usr/local/bin/oracle-v4
RUN --mount=type=cache,id=raster-target,sharing=locked,target=/src/target \
    find src examples tests -type f -exec touch {} + && cargo build --locked --release --example bench && cp target/release/examples/bench /usr/local/bin/bench
# Its own cache mount (the v4 target dir), as in `ci`.
RUN --mount=type=cache,id=raster-target-v4,sharing=locked,target=/src/target \
    find src examples tests -type f -exec touch {} + \
    && RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls" cargo build --locked --release --example bench \
    && cp target/release/examples/bench /usr/local/bin/bench-v4

# PGO and BOLT (pgo.yml, dispatch only; docs/performance.md): cargo-pgo builds the bench example
# instrumented, trains it on the bench page, rebuilds it with the profile, and, when BOLT is
# installed, once more through BOLT. The tools are installed before `COPY . .`, so a new commit
# does not install them again. BOLT comes from apt.llvm.org (bookworm has none of its own); if the
# repository or the package is missing the image still builds and the job reports PGO only.
FROM toolchain AS pgo-tools
RUN (rustup component add llvm-tools || rustup component add llvm-tools-preview) \
    && CARGO_NET_OFFLINE=false cargo install cargo-pgo --locked
RUN apt-get update && apt-get install -y --no-install-recommends wget gnupg \
    && { wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg \
         && echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-19 main" > /etc/apt/sources.list.d/llvm.list \
         && apt-get update && apt-get install -y --no-install-recommends bolt-19 libbolt-19-dev \
         && ln -s /usr/lib/llvm-19/bin/llvm-bolt /usr/local/bin/llvm-bolt \
         && ln -s /usr/lib/llvm-19/bin/merge-fdata /usr/local/bin/merge-fdata \
         && ln -s "$(find /usr/lib/llvm-19 -name libbolt_rt_instr.a | head -n 1)" /usr/local/lib/libbolt_rt_instr.a \
         && test -s /usr/local/lib/libbolt_rt_instr.a; } \
    || echo "BOLT is not available: the PGO job reports PGO only"; \
    rm -rf /var/lib/apt/lists/*

# bench (default build), bench-pgo and, when BOLT worked, bench-bolt (PGO then BOLT), all in
# /usr/local/bin with the Go oracle; tests/parity/pgo.sh times them. The profiles of an earlier run
# stay in the cache mount, so they are deleted first. `bench 40 1` and `bench 40 0` train on the
# single-threaded and the default-threads paths.
FROM pgo-tools AS pgo
COPY . .
COPY --from=go-oracle /out/oracle /usr/local/bin/oracle
RUN --mount=type=cache,id=raster-target-pgo,sharing=locked,target=/src/target \
    find src examples tests -type f -exec touch {} + \
    && rm -rf target/pgo-profiles target/bolt-profiles && find target -name 'bench-bolt-*' -delete \
    && host=$(rustc -vV | sed -n 's/^host: //p') && bin=target/$host/release/examples/bench \
    && cargo build --locked --release --example bench && cp target/release/examples/bench /usr/local/bin/bench \
    && cargo pgo build -- --locked --example bench \
    && cargo pgo run -- --locked --example bench -- 40 1 \
    && cargo pgo run -- --locked --example bench -- 40 0 \
    && cargo pgo optimize build -- --locked --example bench && cp $bin /usr/local/bin/bench-pgo \
    && if command -v llvm-bolt >/dev/null && command -v merge-fdata >/dev/null; then \
         { cargo pgo bolt build --with-pgo -- --locked --example bench \
           && "$(dirname $bin)/bench-bolt-instrumented" 40 1 >/dev/null \
           && cargo pgo bolt optimize --with-pgo -- --locked --example bench \
           && cp "$(find target -type f -name 'bench-bolt-optimized' | head -n 1)" /usr/local/bin/bench-bolt; } \
         || echo "BOLT failed: the job reports PGO only"; \
       else echo "no BOLT: the job reports PGO only"; fi
