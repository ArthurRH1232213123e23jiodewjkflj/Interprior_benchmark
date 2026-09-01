"""Where does our commanded action diverge from the recorded teacher?

Guide tracking now matches zjw (20.8 vs 19.4 mm) but ever_lifted is still 0 while
his cube reaches z=1.059 m. Tracking is driven by the arm, grasping by the hand,
so split the 27 dims and compare against the recording frame by frame. zjw
reports action_mae_arm_rad median 0.057 and action_mae_hand_rad median 0.0096 for
this same case, which gives us a reference for "correct".
"""
import sys

import numpy as np

sys.path.insert(0, "/home/huangsicheng/benchmark")

from interprior_bench.envs.episode import load_derived_episode  # noqa: E402

SHARD = ("/mnt/zuoyufan/lqr_test/runs/20260821/lqr_cube_4096env_1ep_g7_20260821_091345/"
         "runs/lqr_cube_4096env_1ep_g7_20260821_091345_postprocess/derived/success/"
         "shards/derived_000007")
SLOT = 23

states = np.load("/home/huangsicheng/benchmark/train_eval/eval/v_init/replay_states.npz")
ours = states["commanded"][:, 0, :]           # (647, 27) what we commanded
episode = load_derived_episode(SHARD, SLOT, frames=0)
teacher = np.asarray(episode.arrays["robot_joint_pos_target"], dtype=np.float32)[:647]

print(f"ours    {ours.shape}   teacher {teacher.shape}")
delta = np.abs(ours - teacher)
print(f"\naction MAE all  = {delta.mean():.4f} rad   (zjw 0.0244)")
print(f"action MAE arm  = {delta[:, :7].mean():.4f} rad   (zjw 0.0570)")
print(f"action MAE hand = {delta[:, 7:].mean():.4f} rad   (zjw 0.0096)")

print("\n=== per-dim mean |delta| ===")
for index in range(27):
    label = f"arm j{index+1}" if index < 7 else f"hand {index-7}"
    bar = "#" * min(60, int(delta[:, index].mean() * 300))
    print(f"  {label:9s} {delta[:, index].mean():.4f}  {bar}")

print("\n=== does the hand ever close? (teacher vs ours) ===")
for name, series in (("teacher", teacher), ("ours", ours)):
    hand = series[:, 7:]
    span = hand.max(axis=0) - hand.min(axis=0)
    print(f"  {name:8s} hand range: mean {span.mean():.4f} max {span.max():.4f} rad")
    print(f"           at frame 0 / 300 / 646: "
          f"{hand[0].mean():.4f} / {hand[300].mean():.4f} / {hand[646].mean():.4f}")

print("\n=== divergence over time (mean |delta| by segment) ===")
for start in range(0, 647, 100):
    stop = min(start + 100, 647)
    window = delta[start:stop]
    print(f"  frames {start:3d}-{stop:3d}: all {window.mean():.4f}  "
          f"arm {window[:, :7].mean():.4f}  hand {window[:, 7:].mean():.4f}")

rise = states["rise"][:, 0]
print(f"\nour rise: max {rise.max()*1000:.3f} mm at frame {int(np.argmax(rise))}")
