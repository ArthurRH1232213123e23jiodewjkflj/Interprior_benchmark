#!/usr/bin/env bash
# One command: exported authoring json -> runnable benchmark suite.
#
#   bash tasks/build_task/make_task.sh result/my_traj.json my_task_v1
#
# Wraps the two steps you would otherwise run by hand (json -> pointflow npz,
# npz -> suite yaml + reachability verdict) so the frame convention and the
# output locations cannot drift apart.
set -euo pipefail

JSON="${1:?usage: make_task.sh <authoring.json> [suite_name] [-- extra build args]}"
NAME="${2:-authored_$(basename "${JSON%.json}")_v1}"
shift || true; shift || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # tasks/build_task
PKG="$(cd "$HERE/../.." && pwd)"                        # bench_cube_val
ROOT="$(cd "$PKG/.." && pwd)"                           # ~/benchmark
PY="${PYTHON:-python3}"

# Relative json paths are taken from build_task/, so `result/x.json` just works.
[ -f "$JSON" ] || JSON="$HERE/$JSON"
[ -f "$JSON" ] || { echo "no such json: $1" >&2; exit 1; }

FLOWS="$PKG/tasks/cube_reach/flows_authored/$NAME"
mkdir -p "$FLOWS"

echo "[make_task] json   : $JSON"
echo "[make_task] suite  : $NAME"
echo "[make_task] flows  : $FLOWS"
echo

# Step 1 -- json -> goal_pointflow.npz.
#
# --force-robot-base is passed deliberately. The converter otherwise honours the
# `robot_base` recorded INSIDE the json, and the early authoring pages wrote
# [-0.6, 0, 0.6] while the simulator's robot sits at [0, 0, 0.53]. Freshly
# generated pages already declare the right value, so this is a no-op for them and
# a correction for anything older -- either way the npz ends up in the real frame.
echo "[make_task] step 1/2  json -> pointflow npz"
"$PY" "$HERE/goal_traj_to_pointflow.py" \
  --goal-json "$JSON" -K 64 -N 16 \
  --force-robot-base 0,0,0.53 \
  --out-dir "$FLOWS"

# The converter always writes goal_pointflow.npz; the suite builder globs for
# *_pointflow.npz, so give it a name that survives several trajectories per dir.
mv -f "$FLOWS/goal_pointflow.npz" "$FLOWS/${NAME}_pointflow.npz"

echo
echo "[make_task] step 2/2  npz -> suite (+ reachability verdict)"
cd "$ROOT"
"$PY" "$HERE/build_authored_suite.py" \
  --flows "$FLOWS" \
  --pattern '*_pointflow.npz' \
  --name "$NAME" \
  --report-json "$FLOWS/reachability_report.json" \
  "$@"

echo
echo "[make_task] done. If the verdict says OUT, the trajectory left the envelope the"
echo "            teacher data covers -- redraw it, or re-run with --include-out-of-envelope"
echo "            and read the scores as task-infeasible rather than policy failure."
echo
echo "  run it (NVIDIA node only -- Isaac exits 0 after 15 s on the Hygon node):"
echo "    PY=\$HOME/.conda/envs/interprior/bin/python"
echo "    \$PY -m bench_cube_val run \\"
echo "      --policy \"\$PY train_eval/policy_server_exp04.py --strict-finite\" \\"
echo "      --suite bench_cube_val/suites/$NAME.yaml \\"
echo "      --cases 4 --gpus 0 --html-cases 4 \\"
echo "      --interprior-root /mnt/venv_share/H800/Interprior \\"
echo "      --out ~/bench_runs/$NAME"
