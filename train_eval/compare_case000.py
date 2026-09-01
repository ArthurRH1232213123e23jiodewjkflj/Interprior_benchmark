"""Side-by-side of case000: zjw's exp04 eval vs ours, same ckpt and episode.

Guide tracking is comparable (0.0194 vs 0.0180) yet ever_lifted disagrees, so the
question is whether the lift judgement uses a different reference height or the
policy genuinely behaved differently.
"""
import glob
import json

ZJW = json.load(open(
    "/home/zhangjiawei/Interprior_train/runs/evals/"
    "txy_exp04_test200_ckpt_0055000_source121to768_20260824/case000/metrics.json"))
OURS_SUMMARY = json.load(open(sorted(glob.glob(
    "/home/huangsicheng/benchmark/train_eval/eval/exp04_test200_f647/"
    "gpu*/summary.json"))[0]))
OURS = OURS_SUMMARY["cases_detail"][0]

print("=== identity check ===")
print(f"  zjw  episode_uid = {ZJW.get('source_episode_uid')}")
print(f"  ours episode_uid = {OURS.get('episode_uid')}")
print(f"  same episode     = {ZJW.get('source_episode_uid') == OURS.get('episode_uid')}")
print(f"  zjw  shard/slot  = ...{str(ZJW.get('source_shard'))[-16:]} / "
      f"{ZJW.get('source_episode')}")
print(f"  ours shard/slot  = ...{str(OURS.get('shard'))[-16:]} / {OURS.get('slot')}")

KEYS = (
    "lift_threshold_world_m", "initial_object_z_world_m", "ever_lifted",
    "rise_max_m", "rise_end_m",
    "first_lift_source_frame", "first_lift_rollout_frame",
    "lift_longest_frames", "sustained_lift",
    "mean_guide_tracking_error_m", "final_guide_tracking_error_m",
    "guide_tracking_fraction_within_3cm", "guide_tracking_fraction_within_5cm",
    "demo_success", "frames",
)
print(f"\n{'key':42s} {'zjw':24s} ours")
print("-" * 84)
for key in KEYS:
    left = ZJW.get(key, "-")
    right = OURS.get(key, "-")
    flag = ""
    if key in ZJW and key in OURS:
        if isinstance(left, bool) or isinstance(right, bool):
            flag = "   <<<" if left != right else ""
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            denom = max(abs(left), abs(right), 1e-9)
            flag = "   <<<" if abs(left - right) / denom > 0.05 else ""
    print(f"{key:42s} {str(left):24s} {right}{flag}")

print("\n=== every lift/height-related key zjw records ===")
for key in sorted(ZJW):
    if any(word in key for word in ("lift", "rise", "height", "_z_", "grasp")):
        print(f"  {key:46s} = {ZJW[key]!r}")
