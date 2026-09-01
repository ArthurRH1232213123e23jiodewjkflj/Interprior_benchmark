"""Adapter ABC — the policy side of the env/policy boundary.

`TroMpEnv.rollout()` talks to exactly this interface.  Whether the
implementation holds a model in-process (`flow_policy_v10`), replays recorded
teacher targets (`recorded_teacher`), or forwards over a pipe to a server in a
different conda env (`subprocess_adapter`) is invisible to the env.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from ..envs.contract import PolicyContract


class PolicyAdapter(ABC):
    """One evaluated policy.

    Lifecycle, driven by `TroMpEnv.rollout()`:

        handshake()  -> PolicyContract        once, before the env is built
        reset(...)   or  prime(...)           once, after env state injection
        step(obs)    -> model_action          every control tick
        commit_action(action, ...)            after env clipping/IK, same tick
        close()                               once
    """

    # --- handshake ----------------------------------------------------------
    @abstractmethod
    def handshake(self) -> PolicyContract:
        """Declare everything the env branches on. Called before env creation."""

    # --- episode setup ------------------------------------------------------
    def reset(
        self,
        robot_joint_pos: np.ndarray,
        *,
        object_flow_plan: np.ndarray | None = None,
        env_id: int = -1,
    ) -> None:
        """Start from frame 0. Fixed-plan policies get the whole plan here.

        `env_id` addresses one env when a shard runs N cases as N parallel envs;
        each then has its own plan and its own initial joints. -1 (the default)
        means "every env", which is the single-env case. Adapters that hold one
        state per case can ignore it; a shared server must not.
        """

    def prime(
        self,
        *,
        object_flow: np.ndarray,
        robot_joint_pos: np.ndarray,
        previous_action: np.ndarray,
    ) -> None:
        """Start mid-episode: replay history so internal state matches frame N."""

    # --- per-tick -----------------------------------------------------------
    @abstractmethod
    def step(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Return the raw model action for this observation.

        Observation keys, all float32, robot-base frame:
          object_surface_points  (P, 3)   live object surface
          robot_joint_pos        (J,)     live joint positions
          object_flow           (...)     present only when
                                          contract.flow_conditioning == per_step
        """

    def commit_action(
        self,
        model_action: np.ndarray,
        *,
        chunk_invalidation_tolerance_rad: float = 0.0,
    ) -> None:
        """Tell the policy what was *actually* commanded after env safety edits.

        The env clips to joint limits and to a max per-step delta, so the
        executed target can differ from what `step()` returned.  Policies with
        action chunking need this to stay in sync; stateless ones can ignore it.
        """

    def model_action_from_joint_target(self, joint_target: np.ndarray) -> np.ndarray:
        """Invert env-side post-processing back into model action space.

        Default assumes the model action *is* the joint target (joint space).
        """

        return np.asarray(joint_target, dtype=np.float32)

    # --- teardown -----------------------------------------------------------
    def close(self) -> None:
        """Release resources (subprocess, CUDA context, file handles)."""

    # --- diagnostics --------------------------------------------------------
    @property
    def inference_calls(self) -> int:
        """Forward passes so far; reported in metrics.json."""

        return 0

    def stats(self) -> dict[str, Any]:
        """Optional extra fields merged into the report."""

        return {}

    def __enter__(self) -> PolicyAdapter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
