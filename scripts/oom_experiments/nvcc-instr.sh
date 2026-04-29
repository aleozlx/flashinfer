#!/bin/bash
# nvcc-instr.sh — wrap nvcc, record peak (Σ VmRSS over pgroup) per TU
# Install: PATH=/path/to/scripts/oom_experiments:$PATH (must precede /usr/local/cuda/bin)
set -u

REAL_NVCC="${REAL_NVCC:-/usr/local/cuda/bin/nvcc}"
LOG="${NVCC_RSS_LOG:-/tmp/nvcc_per_tu.tsv}"
INTERVAL="${NVCC_SAMPLE_INTERVAL:-0.3}"

# Identify the source file in argv (first .cu/.cpp/.cc encountered)
TU=""
for a in "$@"; do
  case "$a" in
    *.cu|*.cuh|*.cpp|*.cc) TU="$a"; break ;;
  esac
done

# Launch real nvcc in a fresh process group so we can scope sampling
setsid "$REAL_NVCC" "$@" &
PID=$!
PGID=$(awk '{print $5}' /proc/$PID/stat 2>/dev/null || echo $PID)

# Sampler: max-over-time of (sum of VmRSS for processes with this pgid)
peak=0
start=$(date +%s.%N)
while kill -0 "$PID" 2>/dev/null; do
  s=0
  # /proc walk — cheaper than ps and gives both VmRSS and pgrp
  for p in /proc/[0-9]*; do
    [ -r "$p/stat" ] || continue
    pgid_p=$(awk '{print $5}' "$p/stat" 2>/dev/null) || continue
    if [ "$pgid_p" = "$PGID" ]; then
      rss=$(awk '/^VmRSS:/{print $2}' "$p/status" 2>/dev/null || echo 0)
      s=$(( s + ${rss:-0} ))
    fi
  done
  (( s > peak )) && peak=$s
  sleep "$INTERVAL"
done
wait "$PID"; rc=$?
end=$(date +%s.%N)
wall=$(awk -v s="$start" -v e="$end" 'BEGIN{printf "%.3f", e-s}')

# Append row (atomic enough for TSV with low concurrency; switch to flock if needed)
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$(date -u +%FT%TZ)" "${TU:-unknown}" "$peak" "$rc" "$PGID" "$wall" >> "$LOG"

exit "$rc"
