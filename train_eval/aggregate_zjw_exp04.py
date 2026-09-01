"""Aggregate zjw's own exp04 test200 eval, same ckpt and same 200 cases as ours."""
import json
import statistics as stats
from pathlib import Path

D = Path("/home/zhangjiawei/Interprior_train/runs/evals/"
         "txy_exp04_test200_ckpt_0055000_source121to768_20260824")

records = []
for case_dir in sorted(D.glob("case*")):
    path = case_dir / "metrics.json"
    if path.exists():
        records.append(json.loads(path.read_text(encoding="utf-8")))

print(f"cases with metrics: {len(records)}")

BOOLS = ("demo_success", "ever_lifted", "dropped_after_lift", "lift_and_follow",
         "lift_and_hold", "sustained_lift", "trajectory_following",
         "physical_success", "evaluated_full_episode")
print("\n=== boolean tallies ===")
for key in BOOLS:
    present = [r for r in records if key in r]
    if present:
        count = sum(1 for r in present if r[key])
        print(f"  {key:28s} {count:>4d}/{len(present)}")

print("\n=== distributions ===")
for key in ("guide_tracking_mean_m", "mean_guide_tracking_error_m",
            "final_guide_tracking_error_m",
            "guide_tracking_fraction_within_3cm",
            "guide_tracking_fraction_within_5cm",
            "action_mae_all_rad", "action_mae_arm_rad", "action_mae_hand_rad",
            "frames", "final_policy_flow_frames"):
    values = [r[key] for r in records if isinstance(r.get(key), (int, float))]
    if values:
        print(f"  {key:36s} med={stats.median(values):.4f} "
              f"min={min(values):.4f} max={max(values):.4f}")

# Their four-class label, if they record one
for key in ("class", "four_class", "outcome_class"):
    labels = [r[key] for r in records if key in r]
    if labels:
        counts: dict = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        print(f"\n  {key}: {counts}")

frames = {r.get("frames") for r in records}
print(f"\nframes values across cases: {frames}")
print(f"guide thresholds: mean<={records[0].get('guide_mean_success_threshold_m')} "
      f"final<={records[0].get('guide_final_success_threshold_m')}")
