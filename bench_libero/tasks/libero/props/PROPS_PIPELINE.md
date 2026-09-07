# Scene props pipeline — the 11 objects `envs/` never had

`envs/` holds 22 envs, one per **target** object. The 130 LIBERO scenes reference
**33** object categories. The other **11 are never the target** (`target=0` in every
scene), so `build_env_dirs.py` — which builds envs from target objects — skipped them
entirely. They are the fixtures the tasks are *about*.

| category | scenes | asset layout | faces | articulated |
|---|---|---|---|---|
| `wooden_cabinet` | 38 | `articulated_objects/` | 1566 | **baked** |
| `flat_stove` | 34 | `articulated_objects/` | 1038 | **baked** |
| `basket` | 22 | `stable_scanned_objects/` | 66390 | |
| `wine_rack` | 17 | `turbosquid_objects/` | 1179 | |
| `desk_caddy` | 14 | `turbosquid_objects/` | 24970 | |
| `white_cabinet` | 12 | `articulated_objects/` | 1566 | **baked** |
| `cookies` | 10 | `stable_hope_objects/` | 17854 | |
| `glazed_rim_porcelain_ramekin` | 10 | `stable_scanned_objects/` | 16274 | |
| `wooden_tray` | 10 | `turbosquid_objects/` | 180 | |
| `wooden_two_layer_shelf` | 10 | `turbosquid_objects/` | 110 | |
| `microwave` | 6 | `articulated_objects/` | 680 | **baked** |

All 11 are static (`moves≈0`; basket/tray/ramekin each register one 1 cm-threshold
jitter). Nothing was missing from disk — the assets were always in the LIBERO install
on js4. What was missing was the extraction step.

## Why this is not cosmetic

`case 0` is *"close the top drawer of the cabinet and put the black bowl on top of
it"*. Its goal trajectory rises 0.228 m and ends **on the cabinet**. With only the
target object spawned, that endpoint sits in mid air over the table — the support
surface the task is defined against does not exist.

Measured, after adding the props:

```
bowl endpoint  world [0.539, -0.314, 0.758]
cabinet top    world z 0.758
bowl end - cabinet top   = -0.000 m
nearest horizontal cabinet->endpoint = 0.080 m   (inside the cabinet footprint)
```

That agreement is **not fitted**. The cabinet pose comes from LIBERO's `all_poses`,
its mesh from LIBERO's MJCF, the bowl's path from `pointflow/` — three independent
sources landing on 0.000 m. It is the strongest evidence so far that the frame
handling across this whole chain is right.

**28 of 130** trajectories end more than 5 cm above where they start, so 28 cases
have a support surface they depend on.

And the scope is not narrow. Measured over all 130 scenes:

| | scenes |
|---|---|
| contain at least one prop category | **120 / 130** |
| contain none (target object only) | 10 |
| contain a baked-articulated category | 70 |

So this was not an edge case affecting a handful of tasks — **all but 10 of the 130
scenes were being built incomplete.**

## Pipeline

```
LIBERO install (js4 only)   ~/libero/LIBERO/libero/libero/assets/
  │  build_env/extract_libero_props.py --all --out ~/libero_props     [js4, dexverse env]
  ▼
~/libero_props/<category>/mesh.npz  +  props_registry.json
  │  tar pipe through a local machine (js4 cannot reach server2 directly:
  │  Permission denied (publickey) -- same route as the 2026-09-01 move, see MOVED.md)
  ▼
tasks/libero/props/<category>/mesh.npz            verts [N,3] float32 METRES, faces [M,3]
tasks/libero/props/props_registry.json            provenance + geom stats + baked flag
  │  render/libero_task_viewer.py --case N        resolves each scene object from
  ▼                                               envs/ first, then props/
one static HTML with the whole scene
```

Adding a category later is one `--name`, no new code.

## What the extractor has to handle

None of this is uniform across the 11, which is why it is a script and not a one-off.

**Three asset layouts.** `find_xml()` tries them in order:

```
articulated_objects/<name>.xml          assembly referencing part subdirs
<kind>/<dir>/<dir>.xml                  single rigid body
```

`kind` is one of `turbosquid_objects`, `stable_scanned_objects`,
`stable_hope_objects`, `articulated_objects`.

**Naming is not mechanical.** `wooden_cabinet`'s geometry is under
`wooden_cabinet_base` (plain `wooden_cabinet` is the drawer), and `wine_rack` has a
separate `wine_rack_stand`. Resolved through an explicit `DIR_OVERRIDE`, not prefix
guessing. When `articulated_objects/<name>.xml` exists it wins, because the assembly
is more complete than any single part — for the cabinet that adds the drawer, taking
its Y extent from 0.191 to 0.222.

**Three mesh formats.** 8 categories ship `.obj`, `microwave` only `.stl`,
`cookies` and `white_cabinet` only MuJoCo `.msh`. **trimesh cannot read `.msh`**, so
`read_msh()` parses it: 4×int32 header (nvert, nnormal, ntexcoord, nface), then
float32 verts, float32 normals, float32 texcoords, int32 faces. Skipping that format
would have lost 2 of 11.

**Visual geoms are not all meshes.** `flat_stove`'s stove top is `type="box"`.
`primitive_mesh()` generates box/cylinder/sphere/capsule. `is_visual()` keeps
`group="1"` and drops `group="0"` (collision), so collision primitives — the cabinet
has seven — do not leak into the render.

**Scale is per-mesh, from the MJCF, and varies wildly**: 0.0045 (`wine_rack`) to 1.5
(`desk_caddy`). It is read from each `<asset><mesh scale=...>`, never assumed.
Verified independently on the cabinet: raw obj extent 5.13×3.81×4.34 × 0.05 =
0.256×0.191×0.217 m, matching what its collision boxes (already in metres) imply.

**Transforms accumulate down the body tree.** `collect()` walks `<body>` composing
`pos`/`quat`, then applies each geom's own `pos`/`quat`, and merges everything into
one mesh in the object-root frame. That frame is what `all_poses` indexes, so the
recorded pose applies directly.

## Frames — the part that fails silently

`goal_traj` in `pointflow/` is **base frame**; `all_poses` in `flows/` is **world
frame**. To put them together the viewer recovers the offset from the target's own
first pose:

```python
shift = all_poses[target_index, 0, :3] - goal_traj[0, :3]
```

rather than re-deriving a `ROBOT_BASE` constant. This keeps the props aligned with the
target *by construction* — and it is why the cabinet check came out at 0.000 m instead
of merely close. Getting this wrong does not raise; objects float or sink.

## Limits, stated

**Articulation is dropped on purpose.** `flat_stove`, `white_cabinet`, `microwave` and
`wooden_cabinet` are articulated in LIBERO (drawers, doors, knobs). Every visual geom
is baked into one static mesh at the pose the assembly xml declares, recorded as
`articulated_baked: true`. Consequence worth being blunt about: case 0's task is
*"**close the top drawer** and put the bowl on top"* — the first half cannot be scored
against a static prop. **70 of the 130** scenes contain a baked-articulated category.

**`white_cabinet` and `wooden_cabinet` share geometry.** All seven part `.msh` files
are byte-identical (md5-checked per part); LIBERO distinguishes them by texture
(`white_wood.png` vs `dark_fine_wood.png`) alone. Since only geometry is baked, the
two are **indistinguishable in any render**. Tell them apart via `source_xml` in
`props_registry.json`, never by looking. (This tripped me: identical output md5 looked
like an extractor bug until the source files were compared.)

**Decimation is currently inert.** `MAX_FACES=20000` exists, but the `dexverse` env
lacks `fast_simplification`, so `basket` (66390 faces) and `desk_caddy` (24970) went
through whole; the extractor warns and keeps the full mesh rather than failing. Cost:
`basket/mesh.npz` is 3.2 MB, and the 22 scenes using it will produce heavy pages.
Install `fast_simplification` in that env to fix.

**Textures and materials are not carried.** Geometry only; props render in flat grey.

## Verifying a transfer

```bash
PY=~/pro5000_env/.venv_isaacsim_pro5000/bin/python
cd ~/benchmark
$PY - <<'EOF'
import numpy as np, json, os
D = "bench_libero/tasks/libero/props"
reg = json.load(open(f"{D}/props_registry.json"))["props"]
for k in sorted(reg):
    z = np.load(f"{D}/{k}/mesh.npz")
    v, f = z["verts"], z["faces"]
    ok = len(v) == reg[k]["verts"] and len(f) == reg[k]["faces"]
    ok &= bool(np.isfinite(v).all()) and int(f.max()) < len(v)
    print(f"{k:32s} {len(v):7d} {len(f):7d} {'ok' if ok else 'MISMATCH'}")
EOF
```

Counts must match the registry, verts finite, face indices in range. Last run:
11/11 ok. `du -sh` on CFS under-reports these (apparent size) — trust the loader, not
`du`.

## Where the assumption came from

`STRUCTURE.md`'s "22 envs, not 130" reasoning is correct about what it actually
proves: there are only 22 distinct **target** assets. The mistake was equating *env*
with *target object*. An env is a **scene**, and a scene has ~5 objects. That single
conflation is why all 95 cases were built with incomplete scenes, and it survived
because the missing objects are static — nothing errored, the target still moved, and
only the endpoint geometry gave it away.

