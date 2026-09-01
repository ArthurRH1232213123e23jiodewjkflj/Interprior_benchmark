"""Is `mixed_world_and_robot_base` a real frame difference, or just extra arrays?

The min loader asserts coordinate_frame == "robot_base". The 20260821 full shards
declare "mixed_world_and_robot_base". Before relaxing that gate we have to know
whether the arrays the loader actually reads are still robot-base, or whether
"mixed" means the unsuffixed arrays changed meaning. Feeding world-frame poses to
a base-frame policy would produce a plausible, wrong number.
"""
import json

import numpy as np
import zarr

FULL = ("/mnt/zuoyufan/lqr_test/runs/20260821/lqr_cube_4096env_1ep_g7_20260821_091345/"
        "runs/lqr_cube_4096env_1ep_g7_20260821_091345_postprocess/derived/success/"
        "shards/derived_000007")
MIN = ("/mnt/zuoyufan/tro_output/runs/20260818/"
       "4gpu_10k_env_2ep_20260818_174153_batch01_gpu4/runs/"
       "4gpu_10k_env_2ep_20260818_174153_batch01_gpu4_postprocess/derived/success/"
       "shards/derived_000427")

for label, shard, slot in (("FULL 0821", FULL, 23), ("MIN 0818", MIN, 2)):
    commit = json.loads(open(f"{shard}/commit.json").read())
    group = zarr.open_group(f"{shard}/frames.zarr", mode="r")
    print(f"=== {label} ===")
    print(f"  coordinate_frame = {commit.get('coordinate_frame')!r}")
    print(f"  profile          = {commit.get('profile')!r}")

    pose = np.asarray(group["object_root_current_pose"][slot, 0], dtype=np.float64)
    print(f"  object_root_current_pose[0]   = {np.round(pose[:3], 4).tolist()}")
    if "object_root_current_pose_w" in group:
        world = np.asarray(group["object_root_current_pose_w"][slot, 0], dtype=np.float64)
        print(f"  object_root_current_pose_w[0] = {np.round(world[:3], 4).tolist()}")
        offset = world[:3] - pose[:3]
        print(f"  world - base = {np.round(offset, 4).tolist()}")
        print(f"  (robot base at x=-0.6 would show ~[0.6, 0, ...] if base-relative)")
    print(f"  quat = {np.round(pose[3:], 4).tolist()}")
    print()

# The cube sits on a table ~0.6 m up in world z. In robot-base coordinates the
# base is on that table, so base-frame z should be small, world-frame z ~0.6.
print("Read: base-frame z is small (cube near base height); world-frame z ~= table top.")
