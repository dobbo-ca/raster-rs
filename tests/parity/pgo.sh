#!/bin/sh
# Times the bench example built three ways against Go: tests/parity/pgo.sh [N]. Needs `oracle` (Go)
# and `bench` (the default build), `bench-pgo` (cargo-pgo, trained on the bench page) and, when BOLT
# was available, `bench-bolt` (PGO then BOLT) on PATH; run from the repo root (stage `pgo` of
# .github/ci/Dockerfile, workflow pgo.yml). Each build runs tests/parity/bench.sh (its Rust binary
# is $BENCH); the tables give the Rust ms per call of each kernel, single-threaded and then default
# threads, and the change against the default build.
set -eu
export LC_ALL=C
n=${1:-50}
t=$(mktemp -d)
have=
labels=
for v in bench:default bench-pgo:PGO bench-bolt:"PGO + BOLT"; do
  b=${v%%:*}
  if ! command -v "$b" >/dev/null 2>&1; then
    echo >&2 "no $b: skipped"
    continue
  fi
  BENCH=$b sh tests/parity/bench.sh "$n" >"$t/$b.md"
  have="$have $t/$b.md"
  labels="$labels,${v#*:}"
done
[ -n "$have" ] || { echo >&2 "no bench binary on PATH"; exit 1; }
# A kernel row of bench.sh is `| name | Go ms | Rust ms | ratio | Rust 1-thread ms | ratio |`.
# shellcheck disable=SC2086
awk -F'|' -v labels="${labels#,}" '
FNR == 1 { nf++ }
/^[|] [a-z0-9-]+ [|] [0-9]/ {
  name = $2; gsub(/ /, "", name)
  v[nf, 1, name] = $6 + 0
  v[nf, 2, name] = $4 + 0
  if (!(name in seen)) { seen[name] = 1; names[++cnt] = name }
}
END {
  split(labels, lab, ",")
  split("single-threaded,default threads", pass, ",")
  for (p = 1; p <= 2; p++) {
    print "#### Rust ms per call, " pass[p] " (lower is better; the change is against the default build)"
    print ""
    s = "| kernel |"; r = "|---|"
    for (f = 1; f <= nf; f++) { s = s " " lab[f] " |"; r = r "---|"; if (f > 1) { s = s " change |"; r = r "---|" } }
    print s; print r
    for (i = 1; i <= cnt; i++) {
      k = names[i]
      s = "| " k " |"
      for (f = 1; f <= nf; f++) {
        s = s sprintf(" %.3f |", v[f, p, k])
        if (f > 1 && v[1, p, k] > 0) s = s sprintf(" %+.1f%% |", (v[f, p, k] / v[1, p, k] - 1) * 100)
      }
      print s
    }
    print ""
  }
}' $have
for f in $have; do
  echo "<details><summary>$(basename "$f" .md): the full bench.sh table</summary>"
  echo
  cat "$f"
  echo
  echo "</details>"
done
rm -rf "$t"
