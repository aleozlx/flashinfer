#!/bin/bash
# run_experiment.sh — run one experiment cell
# Usage: run_experiment.sh <exp_id> <module_spec> <arches> <max_jobs>
set -euo pipefail

EXP_ID="${1:?exp_id required}"
SPEC="${2:?spec required}"
ARCHES="${3:?arches required, use 'default' to skip override}"
MAX_JOBS="${4:?max_jobs required, use 'auto' for script default}"

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
EXP_DIR="$REPO_ROOT/aot-memory-reports/${EXP_ID}"
mkdir -p "$EXP_DIR"

# Set up nvcc shim
export REAL_NVCC=/usr/local/cuda/bin/nvcc
export NVCC_RSS_LOG="$EXP_DIR/nvcc_per_tu.tsv"
echo -e "timestamp\tsource\tpeak_rss_kb\texit_code\tpgid\twall_s" > "$NVCC_RSS_LOG"

SHIM_DIR="$REPO_ROOT/scripts/oom_experiments"
export PATH="$SHIM_DIR:$PATH"
hash -r
which nvcc
[ "$(which nvcc)" = "$SHIM_DIR/nvcc-instr.sh" ] || { echo "shim not first in PATH"; exit 1; }

# Calibrate MAX_JOBS
if [ "$MAX_JOBS" = "auto" ]; then
  MEM_GB=$(awk '/^MemAvailable:/{print int($2/1024/1024)}' /proc/meminfo)
  NPROC=$(nproc)
  DIV=8   # post-#3204 default; override via env if desired
  MAX_JOBS=$(( MEM_GB / DIV ))
  (( MAX_JOBS < 1 )) && MAX_JOBS=1
  (( NPROC < MAX_JOBS )) && MAX_JOBS=$NPROC
fi
export MAX_JOBS

# Cgroup peak: prefer v2, fallback v1
read_cgroup_peak_kb() {
  if [ -r /sys/fs/cgroup/memory.peak ]; then
    awk '{printf "%d\n", ($1+1023)/1024}' /sys/fs/cgroup/memory.peak
  elif [ -r /sys/fs/cgroup/memory/memory.max_usage_in_bytes ]; then
    awk '{printf "%d\n", ($1+1023)/1024}' /sys/fs/cgroup/memory/memory.max_usage_in_bytes
  else
    echo 0
  fi
}

# Reset cgroup peak (v2 only — write 0)
[ -w /sys/fs/cgroup/memory.peak ] && echo 0 > /sys/fs/cgroup/memory.peak 2>/dev/null || true

# Background sampler for system-wide aggregate
(
  while :; do
    awk -v t="$(date +%s)" '
      /^MemTotal:/{tot=$2}
      /^MemAvailable:/{av=$2}
      END{printf "%s\t%d\t%d\n", t, tot, tot-av}
    ' /proc/meminfo
    sleep 1
  done
) > "$EXP_DIR/meminfo.log" &
SAMPLER_PID=$!
trap "kill $SAMPLER_PID 2>/dev/null || true" EXIT

# Args for build_one_module.py
PY_ARGS=("$SPEC")
[ "$ARCHES" != "default" ] && PY_ARGS+=(--arches "$ARCHES")
PY_ARGS+=(--manifest "$EXP_DIR/manifest.json")

echo "[exp $EXP_ID] spec=$SPEC arches=$ARCHES max_jobs=$MAX_JOBS"
echo "[exp $EXP_ID] cgroup peak before: $(read_cgroup_peak_kb) kB"

set +e
python3 "$SHIM_DIR/build_one_module.py" "${PY_ARGS[@]}" \
  > "$EXP_DIR/build.stdout" 2> "$EXP_DIR/build.stderr"
EC=$?
set -e

CGROUP_PEAK_KB=$(read_cgroup_peak_kb)
echo "[exp $EXP_ID] cgroup peak after:  ${CGROUP_PEAK_KB} kB"
echo "$CGROUP_PEAK_KB" > "$EXP_DIR/cgroup_peak_kb.txt"

# Annotate manifest with runner info + outcome
python3 - <<EOF
import json, os, platform, subprocess
m = json.loads(open("$EXP_DIR/manifest.json").read()) if os.path.exists("$EXP_DIR/manifest.json") else {}
m.update({
    "exp_id": "$EXP_ID",
    "max_jobs": $MAX_JOBS,
    "machine_arch": platform.machine(),
    "hostname": platform.node(),
    "ncpus": os.cpu_count(),
    "cuda_version": subprocess.getoutput("nvcc --version | tail -1") if os.path.exists("/usr/local/cuda/bin/nvcc") else None,
    "exit_code": $EC,
    "cgroup_peak_kb": $CGROUP_PEAK_KB,
})
open("$EXP_DIR/manifest.json","w").write(json.dumps(m, indent=2))
EOF

echo "[exp $EXP_ID] DONE rc=$EC, output in $EXP_DIR"
exit $EC
