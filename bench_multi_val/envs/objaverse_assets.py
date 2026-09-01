"""Per-case Objaverse asset resolution: catalog join + env-cfg pin.

Two halves, matching how the data is actually laid out:

  `resolve_case_asset`  -- reads object_catalog.parquet and builds the metadata
                           dict the pin needs. This has no upstream counterpart:
                           `object_profile` has readers but no writer anywhere in
                           the training repo, so the join is ours to write.
  `pin_episode_object`  -- ported from
                           `eval/runtime/physx_rollout.py:399-441`, which reduces
                           the recorded multi-object pool to one episode's asset
                           BEFORE the env is built.

WHY THE PROFILE CANNOT DO THIS
The physics profile is one file for all 436 cases, but the object differs per
case, so object identity is deliberately absent from it (`object_urdf: ''`). This
module supplies the missing piece per case, and `verify.verify_case_asset_pin`
then asserts it landed -- because the `hasattr` guard below means a checkout
without the pool fields pins *nothing* while every call appears to succeed.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

CATALOG_NAME = "object_catalog.parquet"

# Kept verbatim from upstream: these are the field names the env cfg declares,
# and the pool/legacy split is what lets one function serve both checkouts.
POOL_FIELDS = (
    "object_urdf",
    "object_urdf_pool",
    "object_scale",
    "object_scale_pool",
    "object_bbox_min_m",
    "object_bbox_min_pool_m",
    "object_bbox_max_m",
    "object_bbox_max_pool_m",
    "object_support_height_m",
    "object_support_height_pool_m",
    "cube_pool_size",
    "shuffle_assets",
)


@lru_cache(maxsize=8)
def _catalog(catalog_path: str) -> dict[str, dict[str, Any]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(catalog_path).to_pylist()
    table = {str(row["object_uid"]): row for row in rows}
    if len(table) != len(rows):
        raise ValueError(f"{catalog_path}: object_uid is not unique")
    return table


def catalog_path_for(shard: Path) -> Path:
    """Locate the catalog for a shard.

    Layout is `<root>/<outcome>/shards/<shard>`, so the catalog sits three levels
    up. Walking up is deliberate rather than taking a configured root: it ties
    the asset table to the shard actually being read, so a case cannot be scored
    against another dataset's catalog.
    """

    for parent in Path(shard).resolve().parents:
        candidate = parent / CATALOG_NAME
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no {CATALOG_NAME} above {shard}")


def resolve_case_asset(
    shard: Path,
    object_uid: str,
    *,
    catalog_path: Path | None = None,
    verify_sha256: bool = False,
) -> dict[str, Any]:
    """Build the pin metadata for one case from the object catalog."""

    path = Path(catalog_path) if catalog_path else catalog_path_for(shard)
    row = _catalog(str(path)).get(str(object_uid))
    if row is None:
        raise KeyError(f"object_uid {object_uid!r} absent from {path}")

    urdf = str(row["urdf_uri"])
    # Upstream resolves this as `interprior_root / urdf`, which happens to work
    # only because Path("/a") / "/abs" == "/abs". Assert the absoluteness that
    # coincidence depends on, so a future relative uri fails loudly here instead
    # of silently resolving against the wrong root.
    if not Path(urdf).is_absolute():
        raise ValueError(
            f"catalog urdf_uri is relative ({urdf!r}); callers join it against "
            "interprior_root and would resolve it against the wrong checkout"
        )
    if not Path(urdf).is_file():
        raise FileNotFoundError(f"object urdf missing: {urdf}")

    if verify_sha256:
        import hashlib

        expected = str(row.get("urdf_sha256") or "")
        if expected:
            digest = hashlib.sha256(Path(urdf).read_bytes()).hexdigest()
            if digest != expected:
                raise RuntimeError(
                    f"urdf sha256 differs from catalog for {object_uid}: "
                    f"{digest[:16]} != {expected[:16]}"
                )

    return {
        "object_source": str(row["object_source"]),
        "object_id": str(row["object_id"]),
        "object_uid": str(object_uid),
        "object_urdf": urdf,
        "object_urdf_sha256": row.get("urdf_sha256"),
        "object_profile": {
            "object_scale": [float(v) for v in row["scale_xyz"]],
            "bbox_min_m": [float(v) for v in row["bbox_min_m"]],
            "bbox_max_m": [float(v) for v in row["bbox_max_m"]],
            "reset": {"support_height_m": float(row["support_height_m"])},
        },
        "collider_type": row.get("collider_type"),
        "density_kg_m3": row.get("density_kg_m3"),
        "physics_profile_id": row.get("physics_profile_id"),
        "catalog_path": str(path),
    }


def pin_episode_object(cfg: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    """Reduce a recorded multi-object source pool to this episode's asset.

    Ported from `physx_rollout.py:_pin_episode_object_asset`. Returns the values
    it pinned so the caller can hand them to `verify_case_asset_pin`.

    The `hasattr` guard is upstream's and is kept: it lets the same code run on a
    checkout with pool fields (heterogeneous_pool_mode, one object per env) and
    one without (single global asset). Do not remove it -- but do not trust it
    either, which is what the returned dict is for.
    """

    if metadata.get("object_source") != "objaverse":
        return {"mode": "skipped", "reason": "object_source is not objaverse"}

    profile = metadata.get("object_profile")
    if not isinstance(profile, dict):
        raise ValueError("Objaverse metadata lacks object_profile")

    urdf = str(metadata["object_urdf"])
    scale = tuple(float(v) for v in profile["object_scale"])
    bbox_min = [float(v) for v in profile["bbox_min_m"]]
    bbox_max = [float(v) for v in profile["bbox_max_m"]]
    support = float(profile["reset"]["support_height_m"])

    assets = cfg.assets
    pool_assignments = {
        "object_urdf": "",
        "object_urdf_pool": [urdf],
        "object_scale": None,
        "object_scale_pool": [list(scale)],
        "object_bbox_min_m": None,
        "object_bbox_min_pool_m": [bbox_min],
        "object_bbox_max_m": None,
        "object_bbox_max_pool_m": [bbox_max],
        "object_support_height_m": None,
        "object_support_height_pool_m": [support],
        "cube_pool_size": 1,
        "shuffle_assets": False,
    }
    legacy_assignments = {
        "object_urdf": urdf,
        "object_scale": scale,
        "object_size_m": tuple(
            max(hi - lo, 1.0e-6) for lo, hi in zip(bbox_min, bbox_max)
        ),
        "cube_sampling_mode": "fixed",
        "shuffle_assets": False,
    }

    if all(hasattr(assets, name) for name in POOL_FIELDS):
        assignments, mode = pool_assignments, "pool"
    else:
        missing = [n for n in POOL_FIELDS if not hasattr(assets, n)]
        assignments, mode = legacy_assignments, "legacy"

    for name, value in assignments.items():
        setattr(assets, name, value)

    pinned = {
        "mode": mode,
        "urdf": urdf,
        "scale": list(scale),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "support_height_m": support,
    }
    if mode == "legacy":
        pinned["missing_pool_fields"] = missing
    return pinned


# ---------------------------------------------------------------- n>1 support

POOL_ONLY_FIELDS = (
    "object_urdf_pool",
    "object_scale_pool",
    "object_bbox_min_pool_m",
    "object_bbox_max_pool_m",
    "object_support_height_pool_m",
)

UNIFORM_FIELDS = ("collider_type", "density_kg_m3", "friction")
"""Physics that is per-batch in the env, not per-asset.

`scene_utils` reads `assets.object_collider_type` / `object_friction` /
`object_missing_inertial_density_kg_m3` once for the whole scene, so a batch
mixing two values would silently apply one of them to every object. The upstream
producer refuses such a batch outright
(`render_objflownet_v10_rgbd_worker.py:108`), and so do we."""


def build_object_pool(cfg: Any, metadata_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Pin one object PER ENV for a batch of cases (heterogeneous_pool_mode).

    This is the n>1 counterpart of `pin_episode_object`, and it is what keeps the
    platform's batching advantage on a multi-object benchmark: one Isaac start
    serves the whole shard instead of one per case.

    Requires a checkout that declares the pool fields -- see `scene_utils.py:2000`
    (`heterogeneous_pool_mode`), which is a branch SEPARATE from `cube_pool_mode`
    and therefore not subject to its "object_urdf must be empty when
    cube_sampling_mode='pool'" restriction. Raises rather than degrading, because
    a silent fallback here means every env spawns the same object.

    CALL ORDER: after the profile overlay (the overlay would clobber these) and
    before the env is built. The env-to-pool mapping must then be recovered with
    `align_to_pool_order` once the env exists.
    """

    assets = cfg.assets
    missing = [name for name in POOL_ONLY_FIELDS if not hasattr(assets, name)]
    if missing:
        raise RuntimeError(
            "this checkout cannot spawn one object per env: "
            f"assets lacks {missing}. heterogeneous_pool_mode requires them "
            "(scene_utils.py:2000); the venv_share data-production checkouts "
            "predate it. Use an interprior_root that declares them."
        )

    for name in UNIFORM_FIELDS:
        values = {
            metadata.get(name) for metadata in metadata_list
            if metadata.get(name) is not None
        }
        if len(values) > 1:
            raise ValueError(
                f"one batch cannot mix object {name}: {sorted(values)}. The env "
                "applies a single value scene-wide, so split the batch instead."
            )

    urdfs, scales, bbox_mins, bbox_maxs, supports = [], [], [], [], []
    for metadata in metadata_list:
        profile = metadata["object_profile"]
        urdfs.append(str(metadata["object_urdf"]))
        scales.append(tuple(float(v) for v in profile["object_scale"]))
        bbox_mins.append(tuple(float(v) for v in profile["bbox_min_m"]))
        bbox_maxs.append(tuple(float(v) for v in profile["bbox_max_m"]))
        supports.append(float(profile["reset"]["support_height_m"]))

    # Mirror the upstream producer's assignment exactly: clear every single-asset
    # field, then fill the pools. Leaving object_urdf set makes the env raise
    # "object_urdf and object_urdf_pool are mutually exclusive".
    assets.object_urdf = ""
    assets.object_scale = None
    assets.object_size_m = None
    assets.object_bbox_min_m = None
    assets.object_bbox_max_m = None
    assets.object_support_height_m = None
    assets.object_urdf_pool = tuple(urdfs)
    assets.object_scale_pool = tuple(scales)
    assets.object_bbox_min_pool_m = tuple(bbox_mins)
    assets.object_bbox_max_pool_m = tuple(bbox_maxs)
    assets.object_support_height_pool_m = tuple(supports)

    # Pool index must equal case index, so no shuffling.
    if hasattr(assets, "shuffle_assets"):
        assets.shuffle_assets = False
    if hasattr(assets, "cube_pool_size"):
        assets.cube_pool_size = len(urdfs)

    for name, source in (
        ("object_collider_type", "collider_type"),
        ("object_missing_inertial_density_kg_m3", "density_kg_m3"),
        ("object_friction", "friction"),
    ):
        values = {
            metadata.get(source) for metadata in metadata_list
            if metadata.get(source) is not None
        }
        if values and hasattr(assets, name):
            setattr(assets, name, next(iter(values)))

    return {
        "mode": "pool",
        "pool_size": len(urdfs),
        "urdfs": urdfs,
        "scales": [list(s) for s in scales],
        "bbox_mins": [list(b) for b in bbox_mins],
        "bbox_maxs": [list(b) for b in bbox_maxs],
        "supports": supports,
    }


def align_to_pool_order(env: Any, count: int) -> list[int]:
    """Recover which pool entry each env actually got.

    THIS IS NOT OPTIONAL. `MultiUsdFileCfg` assigns USD paths in source-prim
    order, which is not numeric env order -- `env_10` sorts before `env_2`. So
    env i does not necessarily hold pool entry i. Upstream handles this by
    reordering its episode list (`render_objflownet_v10_rgbd_worker.py:363-374`)
    and asserting the mapping is one-to-one first.

    Skipping this reorders nothing and breaks nothing visibly: every env still
    has an object and the physics gate still passes. It just means case i's
    object-flow plan is sent to the env holding a different object -- the same
    class of defect as sharing one runner across envs, in the asset dimension.

    Returns `indices` such that `case_for_env[i] = cases[indices[i]]`.
    """

    mapping = getattr(env, "_object_asset_index_per_env", None)
    if mapping is None:
        raise RuntimeError(
            "env exposes no _object_asset_index_per_env: the pool did not take "
            "effect, so every env may hold the same object"
        )
    indices = [int(value) for value in mapping.detach().cpu().tolist()]
    if sorted(indices) != list(range(count)):
        raise RuntimeError(
            f"env-to-object pool mapping is not one-to-one: {indices}"
        )
    return indices
