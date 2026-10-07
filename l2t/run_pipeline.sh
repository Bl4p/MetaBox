#!/usr/bin/env bash
# Start (or resume) the full L2T experiment pipeline in the background.
#
#   l2t/run_pipeline.sh                 # all stages, workers = logical cores - 2
#   l2t/run_pipeline.sh 22              # explicit worker count
#   l2t/run_pipeline.sh 22 core         # one stage only (see: python -m l2t.experiments --list)
#
# Finished trainings and evaluations are skipped, so it is safe to stop and restart.
# Progress: tail -f l2t_runs/experiments.log
# Run it from bash (zsh's background-job niceness slowed a run ~50x on the Mac).
set -euo pipefail
cd "$(dirname "$0")/.."

cores=$( (nproc 2>/dev/null || sysctl -n hw.logicalcpu) )
workers="${1:-$((cores > 3 ? cores - 2 : 1))}"
stage="${2:-all}"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python"

mkdir -p l2t_runs
nohup "$PY" -W ignore -u -m l2t.experiments --stage "$stage" --workers "$workers" \
    >> l2t_runs/experiments.log 2>&1 &
pid=$!
echo "pipeline pid $pid, $workers workers, stage '$stage'  (log: l2t_runs/experiments.log)"

# keep a Mac awake while it runs (no-op elsewhere); a closed lid still sleeps
command -v caffeinate >/dev/null 2>&1 && nohup caffeinate -i -s -w "$pid" >/dev/null 2>&1 &
exit 0
