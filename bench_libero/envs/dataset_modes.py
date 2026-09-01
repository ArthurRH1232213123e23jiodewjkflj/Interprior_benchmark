"""Dataset-mode constants and the motion-boundary detector.

Vendored from `flow_policy/datasets` so `_validate_objflownet_motion_start`
needs no training-package import.  Motion-filtered checkpoints were trained
starting at the first frame whose arm target actually moves; evaluating from an
earlier frame feeds the policy a static regime it never saw, which is why this
is a hard gate rather than a warning.
"""

from __future__ import annotations

import numpy as np

MOTION_TO_LIFT = "motion_to_lift"
"""Training window runs from first arm motion to the lift event."""

FULL_EPISODE = "full_episode"

DATASET_MODES = (MOTION_TO_LIFT, FULL_EPISODE)

OBJFLOWNET_TRAINING_MIN = "objflownet_training_min"


def detect_initial_arm_motion_frame(
    arrays: dict[str, np.ndarray],
    *,
    threshold_rad: float,
    target_key: str = "robot_joint_pos_target",
    arm_joints: int = 7,
) -> int:
    """First frame whose arm target moves more than `threshold_rad`.

    Compares consecutive commanded arm targets; the servo floor means recorded
    targets are never bit-identical, so the threshold (not inequality) defines
    motion.
    """

    if target_key not in arrays:
        raise KeyError(f"episode has no array {target_key!r}")
    targets = np.asarray(arrays[target_key], dtype=np.float32)
    if targets.ndim != 2 or targets.shape[0] < 2:
        raise ValueError(
            f"{target_key} must be [T, D] with T>=2; got {targets.shape}"
        )
    deltas = np.abs(np.diff(targets[:, :arm_joints], axis=0)).max(axis=1)
    moving = np.flatnonzero(deltas > threshold_rad)
    if moving.size == 0:
        raise RuntimeError(
            f"episode never exceeds the static arm-target threshold "
            f"{threshold_rad} rad; cannot locate a motion boundary"
        )
    # +1: diff index i compares frames i and i+1, so motion begins at i+1.
    return int(moving[0]) + 1
