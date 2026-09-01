#!/usr/bin/env bash
# exp04 on the 200 held-out LQR cases, sharded across GPUs.
#
# Why this is not `bench run`, two reasons:
#
# 1. api.py passes --frames unconditionally (cli.py defaults it to 0), and the
#    loader reads 0 as "the full 2400-frame recording". The horizon then becomes
#    2400, so the plan slice [121, 121+2400) needs 2521 frames of a 2400-frame
#    array and trips the length guard. The guard is right and the error is right:
#    exp04 trained on source frames [121, 768) with plan_frames=647, so a
#    2400-frame horizon would run 1753 frames past anything it ever saw. What we
#    want is --frames UNSET -> the 768-frame template window, with max_frames: 647
#    capping the horizon, which makes the slice exactly [121, 768) = 647 frames.
#
#    (For the record, since the wrong version of this note was briefly written
#    here: the object flow is NOT 768 frames long. Every array is 2400 frames;
#    768 is just the loader's default window, `v10_frame_window:
#    template_plan_frames`. Measured on episode 2c47ed94: object_surface_points
#    is (2400, 1024, 3) at frames=0 and (768, 1024, 3) at frames=None.)
#
# 2. api.py does not forward --max-target-step-rad, which must be 100.0. The
#    0.05 default clips every single step -- zjw still logged 333 clips at 100.0.
#
# Both are open items in log/Structure.md. Fixing them means changing shared CLI
# defaults, which would move everyone's replay behaviour, so that is deliberately
# not bundled into the run that produces the number. We call the driver directly,
# one process per GPU, and write into gpu<N>/ so api.py's own merge helper can
# still read the shards.
set -uo pipefail

export HOME=/home/huangsicheng
export TMPDIR=$HOME/tmp
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p "$TMPDIR"

BENCH=$HOME/benchmark
ISAAC=$HOME/pro5000_env/.venv_isaacsim_pro5000/bin/python
PY=$HOME/.conda/envs/interprior/bin/python
SUITE=$BENCH/train_eval/suites/exp04_test200_v1.yaml
# 0829 rerun: the 0828 run predates the reference-alignment fix, so its pos_err,
# rot_err and every guide_tracking_* number was measured against source frames
# [0,647) while the policy chased [121,768). Kept on disk for comparison.
OUT=${OUT:-$HOME/bench_runs/exp04_test200_0829}
GPUS=(1 2 3 4 5 7)
TOTAL=200
HTML_CASES=8          # all on the first shard: a page is tens of MB

rm -rf "$OUT"; mkdir -p "$OUT/logs"
cd "$BENCH"

# Contiguous blocks, not round-robin: shard 0 then holds cases 0-33, so the 8
# HTML pages are cases 0-7 and line up with the earlier 8-case run.
n_gpu=${#GPUS[@]}
per=$(( (TOTAL + n_gpu - 1) / n_gpu ))
pids=()
for i in "${!GPUS[@]}"; do
    gpu=${GPUS[$i]}
    start=$(( i * per ))
    [ "$start" -ge "$TOTAL" ] && continue
    end=$(( start + per - 1 ))
    [ "$end" -ge "$TOTAL" ] && end=$(( TOTAL - 1 ))
    indices=$(seq -s, "$start" "$end")
    html=0
    [ "$i" -eq 0 ] && html=$HTML_CASES
    shard_out=$OUT/gpu$gpu
    mkdir -p "$shard_out"
    echo "[launch] gpu$gpu cases $start-$end ($(( end - start + 1 ))) html=$html"
    CUDA_VISIBLE_DEVICES=$gpu nohup $ISAAC -u scripts/bench_replay.py \
        --suite "$SUITE" \
        --interprior-root /mnt/venv_share/H800/Interprior \
        --policy-cmd "$PY $BENCH/train_eval/policy_server_exp04.py --strict-finite --fixed-replan-schedule --replan-interval 30" \
        --case-indices "$indices" \
        --out "$shard_out" \
        --html-cases "$html" \
        --capture-stride 4 \
        --max-target-step-rad 100.0 \
        --chunk-invalidation-tolerance-rad 0.001 \
        > "$OUT/logs/gpu$gpu.log" 2>&1 &
    pids+=($!)
done

echo "[launch] ${#pids[@]} shards started: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do
    wait "$p" || { echo "[launch] pid $p exited nonzero"; fail=$((fail+1)); }
done
echo "[launch] done, $fail shard(s) reported failure"
ls -d "$OUT"/gpu*/summary.json 2>/dev/null | wc -l | xargs echo "[launch] summaries written:"
