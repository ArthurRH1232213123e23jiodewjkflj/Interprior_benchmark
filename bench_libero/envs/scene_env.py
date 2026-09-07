"""A TroMpEnv that also spawns each case's own scene objects.

WHY A SUBCLASS AND NOT AN EDIT. The env lives in a root-owned, read-only
checkout shared by three benchmarks with a frozen physics profile
(/mnt/venv_share/H800/Interprior). `_setup_scene` is an overridable method and
TroMpEnv itself already overrides `_get_dones`/`_get_rewards` precisely so the
parent stays untouched, so we take the same route: call super() unchanged, then
append our bodies.

Registering extra rigid bodies is safe. The 14 references to `env.object`
across reset/action/obs/reward/termination all read or write THAT object; none
iterates the scene's rigid bodies. `scene.rigid_objects` is a plain dict, and
DirectRLEnv refreshes whatever is in it each step.

RESET. `env.object` has reset_utils to place it every episode. Ours do not, so
`_reset_idx` writes their poses after super() -- without that they stay wherever
the spawner dropped them, which looks like objects sunk into the table.
"""

from __future__ import annotations

import gymnasium as gym
import torch

from isaacsimenvs.tasks.tro_mp.tro_mp_env import TroMpEnv

from . import scene_spawn as SP

TASK_ID = "Isaacsimenvs-TroMp-LiberoScene-v0"


class LiberoSceneEnv(TroMpEnv):
    """TroMpEnv plus the non-target dynamic objects of each case.

    The case list is passed through the cfg (`cfg.libero_scene_cases`) rather
    than a constructor argument, because gym.make() builds the env from cfg
    alone. Empty or absent means behave exactly like TroMpEnv.
    """

    def _setup_scene(self) -> None:
        super()._setup_scene()
        cases = list(getattr(self.cfg, "libero_scene_cases", None) or [])
        if not cases:
            self._scene_objects = []
            self._scene_spawn_report = {"spawned": [], "skipped": "no cases given"}
            return
        if len(cases) != self.num_envs:
            raise RuntimeError(
                f"libero_scene_cases has {len(cases)} entries but num_envs is "
                f"{self.num_envs}; env i runs case i, so they must match")
        self._scene_spawn_report = SP.spawn(self, cases)
        self._libero_scene_cases = cases

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        cases = list(getattr(self, "_libero_scene_cases", None) or [])
        if cases and getattr(self, "_scene_objects", None):
            SP.write_reset_poses(self, cases, env_ids=env_ids)


def register() -> str:
    """Register the task id, idempotently. Returns the id."""
    if TASK_ID not in gym.registry:
        gym.register(
            id=TASK_ID,
            entry_point=f"{__name__}:LiberoSceneEnv",
            order_enforce=False,
            disable_env_checker=True,
            kwargs={},
        )
    return TASK_ID


__all__ = ["LiberoSceneEnv", "register", "TASK_ID"]
