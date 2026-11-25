#!/bin/sh
# Go and Rust kernel time on a 2550x3300 page, as a markdown table: tests/parity/bench.sh [N] [ROUNDS].
# Needs `oracle` (Go) and `bench` (Rust) on PATH; the Rust binary is $BENCH (default bench). When
# `oracle-v4` (Go built with GOAMD64=v4) and $BENCH_V4 (default bench-v4, Rust built with
# -C target-cpu=x86-64-v4 and the false-dependency flags of docs/performance.md) are on PATH too, a
# second table "v4 builds" follows, and each kernel's output size must equal the default build's.
# Go raster has no goroutines, so Go at its default parallelism is Go single-threaded: one Go
# configuration. Rust has two: its default threads (`Limits::default`) and 1 thread. The first
# table also has "Rust AVX2 ms (1 thread)": the same bench with RASTER_SIMD=avx2 (the AVX-512 tier
# capped at AVX2) at 1 thread, and the ratio 1-thread default / AVX2, which is AVX-512 / AVX2 on a
# host with AVX-512 (below 1: AVX-512 wins). On a host without AVX-512 both are the AVX2 tier.
set -eu
export LC_ALL=C
n=${1:-20}
rounds=${2:-5}
bench=${BENCH:-bench}
bench4=${BENCH_V4:-bench-v4}
v4=0
# A host without AVX-512 would die with SIGILL on the v4 binaries: time the default builds under the v4 names instead.
if command -v oracle-v4 >/dev/null 2>&1 && command -v "$bench4" >/dev/null 2>&1 && ! grep -qw avx512f /proc/cpuinfo 2>/dev/null; then
  echo "bench: no AVX-512 on this host; the v4 columns time the default builds" >&2
  vdir=$(mktemp -d)
  ln -s "$(command -v oracle)" "$vdir/oracle-v4"
  ln -s "$(command -v "$bench")" "$vdir/$bench4"
  PATH=$vdir:$PATH
fi
if command -v oracle-v4 >/dev/null 2>&1 && command -v "$bench4" >/dev/null 2>&1; then v4=1; fi
# The runners are shared, so one run can be 20-30% off. Go, Rust and Rust 1-thread (and the same
# three for the v4 builds; the first table adds Rust AVX2 at 1 thread) run in turn, `rounds` times, and each column keeps its fastest round per
# kernel (the same rule for all of them).
for f in go rs rs1 rsa gov rsv rsv1; do : >/tmp/$f.all; done
i=0
while [ "$i" -lt "$rounds" ]; do
  oracle -bench "$n" >>/tmp/go.all
  "$bench" "$n" >>/tmp/rs.all
  "$bench" "$n" 1 >>/tmp/rs1.all
  RASTER_SIMD=avx2 "$bench" "$n" 1 >>/tmp/rsa.all
  if [ "$v4" = 1 ]; then
    oracle-v4 -bench "$n" >>/tmp/gov.all
    "$bench4" "$n" >>/tmp/rsv.all
    "$bench4" "$n" 1 >>/tmp/rsv1.all
  fi
  i=$((i + 1))
done
fastest() { awk -F'\t' '!($1 in t) || $3 < t[$1] { t[$1] = $3; b[$1] = $2 } END { for (k in t) printf "%s\t%s\t%s\n", k, b[k], t[k] }' "$1" | sort; }
tab=$(printf '\t')
# table GO RUST RUST1 GO_LABEL RUST_LABEL [RUST_AVX2]: the tsv files are name, bytes, ns. A kernel on
# one side only, or different output sizes, fails the job. With RUST_AVX2 the table has two more
# columns: Rust AVX2 at 1 thread and Rust 1 thread / Rust AVX2.
table() {
  for f in "$2" "$3" ${6:+"$6"}; do
    [ "$(join -t "$tab" -v1 -v2 "$1" "$f" | wc -l)" -eq 0 ] || { echo "kernels differ between Go and Rust" >&2; exit 1; }
  done
  if [ -n "${6:-}" ]; then a=1; else a=0; fi
  { join -t "$tab" "$1" "$2" | join -t "$tab" - "$3" | if [ "$a" = 1 ]; then join -t "$tab" - "$6"; else cat; fi; } | awk -F'\t' -v g="$4" -v r="$5" -v a="$a" '
$2 != $4 || $2 != $6 || (a && $2 != $8) { print "bytes differ for " $1 ": " $2 " vs " $4 " / " $6 (a ? " / " $8 : "") >"/dev/stderr"; bad = 1 }
BEGIN {
  h = "| kernel | " g " ms | " r " ms (default threads) | " r " / " g " | " r " ms (1 thread) | " r " / " g " |"
  d = "|---|---|---|---|---|---|"
  if (a) { h = h " " r " AVX2 ms (1 thread) | AVX-512 / AVX2 |"; d = d "---|---|" }
  print h
  print d
}
{
  line = sprintf("| %s | %.3f | %.3f | %.2f | %.3f | %.2f |", $1, $3 / 1e6, $5 / 1e6, $5 / $3, $7 / 1e6, $7 / $3)
  if (a) line = line sprintf(" %.3f | %.2f |", $9 / 1e6, $7 / $9)
  print line
}
END { exit bad }'
}
fastest /tmp/go.all >/tmp/go.tsv
fastest /tmp/rs.all >/tmp/rs.tsv
fastest /tmp/rs1.all >/tmp/rs1.tsv
fastest /tmp/rsa.all >/tmp/rsa.tsv
echo "nproc: $(nproc)"
echo
table /tmp/go.tsv /tmp/rs.tsv /tmp/rs1.tsv Go Rust /tmp/rsa.tsv
if [ "$v4" = 1 ]; then
  fastest /tmp/gov.all >/tmp/gov.tsv
  fastest /tmp/rsv.all >/tmp/rsv.tsv
  fastest /tmp/rsv1.all >/tmp/rsv1.tsv
  # Sanity check on the bench output only: the second column is the destination size, fixed per kernel,
  # so this catches a changed row set, not a miscompile (the v4 unit and golden tests in `ci` do that).
  [ "$(cut -f1,2 /tmp/go.tsv)" = "$(cut -f1,2 /tmp/rsv.tsv)" ] || { echo "v4 output sizes differ from the default build" >&2; exit 1; }
  echo
  echo '#### v4 builds (Go GOAMD64=v4, Rust -C target-cpu=x86-64-v4 with the false-deps features)'
  echo
  table /tmp/gov.tsv /tmp/rsv.tsv /tmp/rsv1.tsv "Go v4" "Rust v4"
fi
