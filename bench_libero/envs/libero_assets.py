"""Pin one LIBERO object onto the env config, per case.

The counterpart of `bench_multi_val/envs/objaverse_assets.py`, with the catalog
replaced by `tasks/libero/env_registry.json` and the asset being a generated
URDF next to each env's mesh (`build_env/build_object_urdf.py` explains why a
URDF and not the DexVerse USD).

Two failure modes this exists to prevent, both silent:

* An unpinned run spawns the profile's default `cube_0p06m.urdf`, so every
  LIBERO case would score a 6 cm cube following a bottle's path -- with green
  physics gates. `verify_case_pin` reads the cfg back and refuses that.
* `object_scale` must be set whenever `object_urdf` is (`scene_utils.py:1934`
  raises otherwise), and `object_size_m` feeds the reward's size normalisation.
  Pinning the URDF alone would either crash late or normalise against the cube.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REGISTRY_NAME = "env_registry.json"

#: Fields the pool-capable checkouts add. Mirrors objaverse_assets.POOL_FIELDS:
#: present -> per-env assets are possible; absent -> one global object.
POOL_FIELDS = (
    "object_urdf_pool",
    "object_scale_pool",
    "object_bbox_min_pool_m",
    "object_bbox_max_pool_m",
    "object_support_height_pool_m",
)


def registry_path_for(envs_dir: str | Path) -> Path:
    """Locate env_registry.json for an envs directory (it sits beside it)."""
    envs_dir = Path(envs_dir).resolve()
    candidate = envs_dir.parent / REGISTRY_NAME
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"no {REGISTRY_NAME} beside {envs_dir}")


def _registry(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    table = {str(r["env_id"]): r for r in payload["envs"]}
    if len(table) != len(payload["envs"]):
        raise ValueError(f"{path}: env_id is not unique")
    return table


def resolve_env_asset(
    envs_dir: str | Path,
    env_id: str,
    *,
    verify_sha256: bool = True,
) -> dict[str, Any]:
    """Build the pin metadata for one LIBERO env."""
    envs_dir = Path(envs_dir).resolve()
    row = _registry(registry_path_for(envs_dir)).get(str(env_id))
    if row is None:
        raise KeyError(f"env_id {env_id!r} absent from {registry_path_for(envs_dir)}")

    env_path = envs_dir / env_id
    urdf = env_path / f"{env_id}.urdf"
    if not urdf.is_file():
        raise FileNotFoundError(
            f"{env_id}: no URDF at {urdf}. Run build_env/build_object_urdf.py."
        )
    # The env joins this against interprior_root; that join only leaves the path
    # alone because Path("/a") / "/abs" == "/abs". Assert what it relies on.
    if not urdf.is_absolute():
        raise ValueError(f"{urdf} must be absolute")

    mesh = env_path / row["mesh"]
    if verify_sha256:
        digest = hashlib.sha256(mesh.read_bytes()).hexdigest()
        if digest != row["mesh_sha256"]:
            raise RuntimeError(
                f"{env_id}: mesh.npz sha256 differs from the registry "
                f"({digest[:16]} != {row['mesh_sha256'][:16]}) -- the asset changed "
                "under a frozen index; regenerate the registry deliberately"
            )

    bbox_min = [float(v) for v in row["bbox_min_m"]]
    bbox_max = [float(v) for v in row["bbox_max_m"]]
    # The object rests on the table, and the pointflow pipeline already rebased
    # every task's support height into goal_traj (measured: base-frame z0 is
    # 0.005-0.046 m for floor, low_table AND table alike). So the support height
    # is the table, not the LIBERO scene's surface.
    return {
        "object_source": "libero",
        "env_id": str(env_id),
        "object_urdf": str(urdf),
        "object_urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "mesh_sha256": row["mesh_sha256"],
        "object_profile": {
            "object_scale": [1.0, 1.0, 1.0],
            "bbox_min_m": bbox_min,
            "bbox_max_m": bbox_max,
        },
        "extent_m": float(row["extent_m"]),
        "oversize": bool(row.get("oversize", False)),
        "registry_path": str(registry_path_for(envs_dir)),
    }


def pin_case_object(cfg: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    """Point the env config at this case's object. Returns what it pinned."""
    if metadata.get("object_source") != "libero":
        return {"mode": "skipped", "reason": "object_source is not libero"}

    profile = metadata["object_profile"]
    urdf = str(metadata["object_urdf"])
    scale = tuple(float(v) for v in profile["object_scale"])
    bbox_min = [float(v) for v in profile["bbox_min_m"]]
    bbox_max = [float(v) for v in profile["bbox_max_m"]]
    size = tuple(max(hi - lo, 1.0e-6) for lo, hi in zip(bbox_min, bbox_max))

    assets = cfg.assets
    # `cube_sampling_mode='pool'` REFUSES a non-empty object_urdf
    # (scene_utils.py:1893), so the single-object branch is the one to use, and
    # it needs cube_sampling_mode back to 'fixed'.
    assignments = {
        "object_urdf": urdf,
        "object_scale": scale,
        "object_size_m": size,
        "cube_sampling_mode": "fixed",
        "shuffle_assets": False,
    }
    for name in POOL_FIELDS:
        if hasattr(assets, name):
            assignments[name] = None

    for name, value in assignments.items():
        if hasattr(assets, name):
            setattr(assets, name, value)

    return {
        "mode": "single",
        "urdf": urdf,
        "scale": list(scale),
        "size_m": list(size),
        "env_id": metadata["env_id"],
        "pool_fields_present": [n for n in POOL_FIELDS if hasattr(assets, n)],
    }


def verify_case_pin(cfg: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    """Read the cfg back and prove the intended object is what will spawn.

    The gate that matters. Without it a pin that silently no-ops leaves the
    profile's `cube_0p06m.urdf` in place and the run reports plausible numbers
    for the wrong object -- the shape of the 436-case bug in log/0829 and of the
    xhand run that evaluated a cube instead of the trained mesh.
    """
    problems: list[str] = []
    assets = cfg.assets
    want = str(metadata["object_urdf"])
    got = str(getattr(assets, "object_urdf", "") or "")

    if not got:
        problems.append("assets.object_urdf is empty: the env would spawn its default object")
    elif Path(got).resolve() != Path(want).resolve():
        problems.append(f"assets.object_urdf is {got!r}, expected {want!r}")
    if Path(got).name.startswith("cube_"):
        problems.append(f"assets.object_urdf is still a cube ({Path(got).name})")

    scale = getattr(assets, "object_scale", None)
    if scale is None:
        problems.append("assets.object_scale is None: scene_utils raises when object_urdf is set")

    mode = str(getattr(assets, "cube_sampling_mode", "fixed"))
    if mode == "pool":
        problems.append("cube_sampling_mode='pool' rejects a non-empty object_urdf")

    for name in POOL_FIELDS:
        value = getattr(assets, name, None)
        if value:
            problems.append(f"{name} is still populated: a pool would override the single asset")

    size = getattr(assets, "object_size_m", None)
    if size is not None:
        expected = [
            max(hi - lo, 1.0e-6)
            for lo, hi in zip(metadata["object_profile"]["bbox_min_m"],
                              metadata["object_profile"]["bbox_max_m"])
        ]
        if max(abs(float(a) - float(b)) for a, b in zip(size, expected)) > 1e-6:
            problems.append(f"object_size_m {list(size)} != bbox extent {expected}")

    return {
        "ok": not problems,
        "problems": problems,
        "env_id": metadata.get("env_id"),
        "object_urdf": got,
        "checked": ["object_urdf", "object_scale", "object_size_m",
                    "cube_sampling_mode", *POOL_FIELDS],
    }
