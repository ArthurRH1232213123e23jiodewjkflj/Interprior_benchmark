"""Recorded-teacher adapter — replays the shard's own joint targets.

Upstream this was the `--diagnostic-teacher-actions` branch, an `if runner is
None` fork threaded through the main loop.  As an adapter it stops being a
special case: the env runs its normal loop and just happens to be driven by
recorded actions.

This is the `bench replay` gate for physics alignment.  Reproducing the known
baseline (pos err 1.3 mm, `lift_and_follow=True` on
`derived_000203_slot3`) means the extracted env steps PhysX exactly as the
training stack did.

Note `limit_delta`: recorded targets are ground truth and must reach PhysX
unmodified, so `TroMpEnv.clip_target(..., limit_delta=True)` would corrupt the
comparison.  The env applies the max-step clamp only to policy actions; teacher
replay therefore reports zero safety corrections when alignment is correct.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..envs.contract import (
    ACTION_SPACE_JOINT,
    NOT_APPLICABLE,
    PolicyContract,
)
from ..envs.episode import load_derived_episode
from .base import PolicyAdapter


class RecordedTeacherAdapter(PolicyAdapter):
    """Emit `robot_joint_pos_target[frame]` from the derived shard."""

    def __init__(
        self,
        shard: str | Path,
        slot: int = 0,
        *,
        start_frame: int = 0,
        target_array: str = "robot_joint_pos_target",
        frames: int | None = None,
    ) -> None:
        self.shard = Path(shard)
        self.slot = slot
        self.start_frame = start_frame
        self.target_array = target_array
        self._episode = load_derived_episode(self.shard, self.slot, frames=frames)
        if target_array not in self._episode.arrays:
            raise KeyError(f"episode has no array {target_array!r}")
        self._targets = np.asarray(
            self._episode.arrays[target_array], dtype=np.float32
        )
        self._cursor = start_frame
        self._calls = 0

    # --- handshake ----------------------------------------------------------
    def handshake(self) -> PolicyContract:
        """Joint space, consumes no flow, no evaluation-source gates.

        Teacher replay declares the minimum: NOT_APPLICABLE for both flow
        fields (the sentinel metrics.json already uses for this mode) and None
        for every optional gate, so the env skips those branches entirely.
        """

        return PolicyContract(
            name=f"recorded_teacher:{self.shard.name}/slot{self.slot}",
            action_space=ACTION_SPACE_JOINT,
            action_dim=int(self._targets.shape[1]),
            flow_frames=0,
            flow_conditioning=NOT_APPLICABLE,
            flow_representation=NOT_APPLICABLE,
        )

    # --- episode setup ------------------------------------------------------
    def reset(
        self,
        robot_joint_pos: np.ndarray,
        *,
        object_flow_plan: np.ndarray | None = None,
        env_id: int = -1,
    ) -> None:
        # `env_id` is accepted and ignored: there is one of these adapters PER
        # case, each holding its own cursor, so per-env addressing is already
        # satisfied by construction. Only a shared server needs to route on it.
        del env_id
        self._cursor = self.start_frame
        self._calls = 0

    def prime(
        self,
        *,
        object_flow: np.ndarray,
        robot_joint_pos: np.ndarray,
        previous_action: np.ndarray,
    ) -> None:
        """Mid-episode start: the cursor is the only state to advance."""

        self._cursor = self.start_frame
        self._calls = 0

    # --- per-tick -----------------------------------------------------------
    def step(self, observation: dict[str, np.ndarray] | None = None) -> np.ndarray:
        """Ignore the observation; open-loop by construction."""

        index = min(self._cursor, self._targets.shape[0] - 1)
        self._cursor += 1
        self._calls += 1
        return self._targets[index].copy()

    def commit_action(
        self,
        model_action: np.ndarray,
        *,
        chunk_invalidation_tolerance_rad: float = 0.0,
    ) -> None:
        """Stateless — nothing to reconcile."""

    # --- diagnostics --------------------------------------------------------
    @property
    def inference_calls(self) -> int:
        return self._calls

    def stats(self) -> dict[str, Any]:
        return {
            "action_source": "recorded_teacher",
            "shard": str(self.shard),
            "slot": self.slot,
            "target_array": self.target_array,
            "recorded_frames": int(self._targets.shape[0]),
        }
