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

# ---------------------------------------------------------------- scene table
# A LIBERO scene holds 3-8 objects and this env spawns one (`object_urdf` is a
# single value; the scene registers the fixed keys table/object/goal_viz --
# scene_utils.py:2165-2168). The furniture the goals END ON was therefore
# absent: case 0's bowl was asked to finish on a cabinet top with no cabinet in
# the world, 0.2282 m above the table.
#
# The way in is that the scene already HAS a static body -- the table, baked
# `kinematic_enabled=True, disable_gravity=True` (scene_utils.py:2030) and
# spawned from a plain path the frozen profile invites you to change ("Replace
# this path to use another table URDF"). A URDF holds many links, so the props
# ride along as extra fixed-joint links. No cfg field, no env edit.
#
# THE HAZARD. That same file sets the ROBOT BASE:
# _table_urdf_geometry_center_and_top (scene_utils.py:1560) unions every
# <collision> PRIMITIVE and returns its centre XY and top Z. Props therefore use
# <mesh> collision, which that parser skips (:1592), and unrotated collision
# origins, because it raises on a rotated one before it even looks at the
# geometry type (:1577).
#
# `pin_case_table` swaps the path; `verify_case_table` proves the swap took AND
# that the robot base is unmoved -- by calling the env's own parser, lifted out
# of scene_utils.py with ast so it cannot drift from the real implementation.
# Same shape as pin_case_object/verify_case_pin above and for the same reason:
# a pin that silently no-ops reports plausible numbers for the wrong scene.

def _env_table_parser(interprior_root: str):
    """The env's own _table_urdf_geometry_center_and_top, no reimplementation."""
    import ast
    import types
    import xml.etree.ElementTree as ET

    su = (Path(interprior_root) / "isaacsimenvs" / "tasks" / "simtoolreal"
          / "utils" / "scene_utils.py")
    if not su.exists():
        raise FileNotFoundError(f"env scene_utils.py not found at {su}")
    fn = next(n for n in ast.parse(su.read_text()).body
              if isinstance(n, ast.FunctionDef)
              and n.name == "_table_urdf_geometry_center_and_top")
    mod = types.ModuleType("_su_shim")
    mod.__dict__["ET"] = ET
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<shim>", "exec"), mod.__dict__)
    return mod.__dict__["_table_urdf_geometry_center_and_top"]


def pin_case_table(cfg: Any, composite_urdf: str) -> dict[str, Any]:
    """Point assets.table_urdf at a composite table carrying this case's props."""
    stock = str(getattr(cfg.assets, "table_urdf", "") or "")
    cfg.assets.table_urdf = str(composite_urdf)
    return {"mode": "composite", "table_urdf": str(composite_urdf), "stock_table_urdf": stock}


def verify_case_table(cfg: Any, composite_urdf: str, stock_urdf: str,
                      interprior_root: str) -> dict[str, Any]:
    """Prove the composite is pinned and the robot base did not move."""
    problems: list[str] = []
    got = str(getattr(cfg.assets, "table_urdf", "") or "")
    if not got:
        problems.append("assets.table_urdf is empty")
    elif Path(got).resolve() != Path(composite_urdf).resolve():
        problems.append(f"assets.table_urdf is {got!r}, expected {composite_urdf!r}")

    parse = _env_table_parser(interprior_root)
    ref = parse(str(stock_urdf))
    live = parse(str(got)) if got else None
    if live is None:
        problems.append("could not parse the pinned table URDF")
    else:
        # Bit-identical, not approximately: a moved base invalidates every
        # trajectory in the suite, and the props contribute nothing to these
        # bounds by construction (mesh collision).
        if any(a != b for a, b in zip(ref, live)):
            problems.append(
                f"robot base would MOVE: stock center/top={ref} "
                f"composite={live}; props must use <mesh> collision only")

    import xml.etree.ElementTree as ET
    root = ET.parse(str(got)).getroot()
    for coll in root.findall(".//collision"):
        o = coll.find("origin")
        rpy = (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()
        if any(abs(float(v)) > 1.0e-9 for v in rpy):
            problems.append("a collision origin is rotated; the env parser raises on that")
            break

    n_links = len(list(root.iter("link")))
    n_joints = len(list(root.iter("joint")))
    return {"ok": not problems, "problems": problems,
            "stock_center_top": list(ref), "composite_center_top": list(live) if live else None,
            "n_links": n_links, "n_props": n_joints}
