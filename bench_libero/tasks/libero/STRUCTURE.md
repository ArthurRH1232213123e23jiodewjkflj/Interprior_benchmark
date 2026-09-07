# `tasks/libero/` — structure

LIBERO object-flow benchmark assets inside `bench_libero`. **22 target envs**
(21 distinct geometries) + **11 scene props**, **130 trajectories**,
**95 runnable cases**.

The split that organises everything here: an **env is an object asset** (a directory),
a **case is a trajectory** (a row in `cases.json`). That matches how `bench_cube_val`
and `bench_multi_val` are laid out — 436 objaverse cases are likewise one catalog
plus per-case pins, not 436 directories.

> **Corrected 2026-09-06 — an env is a target object, but a scene is not one object.**
> Everything below about "22 not 130" is still right about *target* assets. But the
> 130 scenes reference **33** categories: the other 11 are never the target, so they
> were never built, and **120 of the 130 scenes were being assembled with objects
> missing** — including the support surfaces goals end on. `case 0`'s bowl is placed
> *on top of a cabinet* that did not exist; its goal endpoint sat 0.228 m in mid air.
> Fixed by `props/`, see **`props/PROPS_PIPELINE.md`**. With the cabinet present the
> bowl's endpoint lands on its top surface to **0.000 m**.

## Layout

```
tasks/libero/
├── envs/                       48M    ← THE ENVS. 22 dirs, generated
│   └── <env_id>/                        e.g. akita_black_bowl, wine_bottle
│       ├── mesh.npz                   HOPE scan: verts/faces/uv. verts already in METRES
│       ├── points_obj.npy             16 canonical surface points, float32 [16,3]
│       ├── env.yaml                   extent/bbox/sha256 + which LIBERO uids & scenes map here
│       ├── <env_id>.obj               spawnable mesh, written by build_env/build_object_urdf.py
│       └── <env_id>.urdf              spawnable asset; references <env_id>.obj by name
│                                      22/22 carry all five. ONE EXCEPTION: alphabet_soup/
│                                      holds 7 extra leftovers from the 0813 single-object
│                                      worked example (.usd, config.yaml, textured.obj,
│                                      textured.mtl, texture_map.png, textures/, usd_src/).
│                                      Those are residue, NOT a second env contract — do not
│                                      copy that shape when adding an env.
├── props/                      6.2M   ← THE OTHER 11. static scene fixtures
│   ├── <category>/mesh.npz              verts [N,3] float32 METRES, faces [M,3]
│   ├── props_registry.json              source_xml + geom stats + articulated_baked
│   └── PROPS_PIPELINE.md                how they are extracted; READ THIS FIRST
├── env_registry.json           24K    all 22 env records + geometry_aliases
├── cases.json                  46K    95 runnable: case_index, shard, slot, goal_npz, env_id, episode_uid
├── reachability_report.json    7.6K   why the other 35 are excluded, per trajectory
├── ENV_LIST.csv                36K    SOURCE OF TRUTH for the 130 rows (usable/obj/suite/stem/T/disp/z0/scene)
│
├── flows/                     4.0M    130 x <stem>_flow.npz + 130 x .done + manifest{,_remote}.json
│                                      obj_traj [T,7] WORLD frame wxyz, all_poses [O,T,7], obj_names, meta
├── pointflow/                 116M    130 x <stem>_flow/ — LEFT AS FOUND, not deduped
│   └── <stem>_flow/                   goal_pointflow.npz: goal_traj [K,7] BASE frame,
│       ├── goal_pointflow.npz         points_obj [16,3], goal_points [K,16,3], flow [K,16,3]
│       └── <obj>_mesh.npz             the 22 meshes copied 130 times = 106M of the 116M
├── viewers/                   930M    54 of 130 self-contained Three.js HTML  ← INCOMPLETE
├── pipeline/                  1.2M    7 extraction/viewer scripts + texture_map.png (run on js4)
├── samples/                    19M    alphabet_soup single-object worked example
│
├── build_env/                  100K   ← THE CODE
│   ├── build_env_dirs.py              generates envs/ + the 3 JSON indexes. Refuses to overwrite.
│   ├── extract_libero_props.py        builds props/ from the LIBERO install. RUNS ON js4.
│   ├── build_object_urdf.py           writes <env_id>.obj + .urdf into each env dir
│   ├── verify_env_dirs.py             22 checks incl. cross-package no-damage. Exit on any failure.
│   ├── verify_object_urdf.py          checks the generated urdf/obj pairs
│   ├── pick_case.py                   case selection helper
│   └── README.md                      the measurements behind "22 not 130"
│
├── MOVED.md                           transfer record from js4 (2026-09-01)
├── README.md / PIPELINE.md / PIPELINE_details.md / task_list.md / 0813_libero.md
```

Consumed by `../../suites/libero_object_flow_v1.yaml` (`goal_source: authored_npz`,
`envs_dir: tasks/libero/envs`, `cases_from: ../tasks/libero/cases.json`).

## Data flow

```
LIBERO demo (HDF5)
  │  pipeline/extract_libero_object_flow.py        [libero env, js4, needs EGL libs]
  ▼
flows/<stem>_flow.npz          obj_traj [T,7]  WORLD frame
  │  pipeline/libero_object_to_pointflow.py       [dexverse env]
  │    - picks target object by largest centroid displacement
  │    - samples 16 surface points from the HOPE mesh (x0.01: cm -> m)
  │    - REBASES to robot base, absorbing the support-surface height
  ▼
pointflow/<stem>_flow/goal_pointflow.npz   goal_traj [K,7]  BASE frame
  │  build_env/build_env_dirs.py                  [any python + numpy]
  ▼
envs/<env_id>/{mesh.npz, points_obj.npy, env.yaml, .obj, .urdf}
  +  cases.json  +  env_registry.json
  │  suites/libero_object_flow_v1.yaml
  ▼
driver.py  ← WIRED (2026-09-01 20:52). Never rolled out; see "Status".
```

## Why 22 envs and not 130

> Read with the correction at the top of this file: this section is about **target**
> assets, and is correct about them. It is *not* a claim that a scene contains one
> object. The 11 non-target categories live in `props/`.

130 is the trajectory count. What a sim needs to spawn an env is the object asset.
Three measurements, none assumed:

**1. Support height is already normalised away.** The pointflow step rebases each
task's support surface into `goal_traj`:

| support | n | world z0 (mean) | base-frame z0 (mean) |
|---|---|---|---|
| floor | 10 | 0.0476 | 0.0464 |
| low_table | 32 | 0.4712 | 0.0247 |
| table | 88 | 0.9133 | 0.0053 |

0.87 m apart in world frame; all within 0.005–0.046 m of the table in base frame.
`world_z - base_z` spans −0.005 → 1.126 (std 0.29), so this is **not** one constant
robot_base being subtracted — the per-task surface height really is absorbed. One
table height serves all 130, and support/z0 are not env axes.

**2. One asset per name.** 22 mesh basenames, each with exactly **one** sha256 across
all 130 tasks (0 basenames carry multiple hashes). `points_obj` is bit-identical
across every task sharing an env — 43 of them for the bowl. Surface points depend
on the asset, not the trajectory.

**3. Suffixed uids are scene instances.** 26 uids collapse to 22 assets:
`akita_black_bowl_{1,2,3}`, `butter_{1,2}`, `yellow_book_{1,2}` each share one mesh.
`new_salad_dressing` and `salad_dressing` are byte-identical → 21 real geometries
(recorded as `geometry_aliases`).

Rejected: **130 dirs** — 79 would be one env with a different path, and `pointflow/`'s
existing 130-way mesh duplication already costs 106M of its 116M; **51** `(object,
support, z0)` combos — two of three axes are fake per table 1; **23** bddl scenes —
cuts across assets, one scene holds several objects.

## Envs

| env_id | extent (m) | vs cube 0.06 | tasks | runnable | uids | scenes | |
|---|---|---|---|---|---|---|---|
| `akita_black_bowl` | 0.1105 | 1.84x | 43 | 29 | 3 | 8 |  |
| `black_book` | 0.1377 | 2.29x | 12 | 12 | 1 | 4 |  |
| `chefmate_8_frypan` | 0.3544 | 5.91x | 7 | 6 | 1 | 2 | **OVERSIZE** |
| `alphabet_soup` | 0.0827 | 1.38x | 6 | 5 | 1 | 4 |  |
| `white_yellow_mug` | 0.1306 | 2.18x | 6 | 6 | 1 | 3 |  |
| `butter` | 0.0772 | 1.29x | 5 | 3 | 2 | 4 |  |
| `chocolate_pudding` | 0.0829 | 1.38x | 5 | 3 | 1 | 4 |  |
| `cream_cheese` | 0.0822 | 1.37x | 5 | 5 | 1 | 5 |  |
| `moka_pot` | 0.1333 | 2.22x | 5 | 4 | 1 | 2 |  |
| `white_bowl` | 0.0832 | 1.39x | 5 | 3 | 1 | 2 |  |
| `ketchup` | 0.1474 | 2.46x | 4 | 1 | 1 | 4 |  |
| `porcelain_mug` | 0.1274 | 2.12x | 4 | 2 | 1 | 3 |  |
| `red_coffee_mug` | 0.1322 | 2.20x | 4 | 1 | 1 | 3 |  |
| `tomato_sauce` | 0.0815 | 1.36x | 4 | 4 | 1 | 4 |  |
| `wine_bottle` | 0.1498 | 2.50x | 4 | 3 | 1 | 2 |  |
| `yellow_book` | 0.1224 | 2.04x | 3 | 2 | 2 | 1 |  |
| `milk` | 0.1398 | 2.33x | 2 | 1 | 1 | 2 |  |
| `orange_juice` | 0.1432 | 2.39x | 2 | 2 | 1 | 2 |  |
| `bbq_sauce` | 0.1093 | 1.82x | 1 | 1 | 1 | 1 |  |
| `new_salad_dressing` | 0.1466 | 2.44x | 1 | 0 | 1 | 1 |  |
| `plate` | 0.1346 | 2.24x | 1 | 1 | 1 | 1 |  |
| `salad_dressing` | 0.1466 | 2.44x | 1 | 1 | 1 | 1 |  |

`chefmate_8_frypan` is 0.354 m — **5.9x the cube**. Nothing the Wuji hand has trained
on is near it; its 7 tasks are a separate question, not part of a headline number.

## cases.json: 95 of 130

Excluded **35**: **12 no_motion** (`ENV_LIST.csv` marks them — the target never
moves, so there is nothing to follow) and **23 xy out of envelope**.

Envelope = the cube donor's measured reach (`bench_cube_val/tasks/build_task`):
xy ≤ 0.687 m, z ≤ 0.653 m, 3D ≤ 0.907 m. LIBERO fits z **130/130** and 3D
**130/130**, but xy only **107/130** — LIBERO reaches further sideways than
the xArm7 does. Same shape as the authored cube trajectories (4/7 usable): outside
the envelope is a task out of distribution, not a policy failure. The bowl loses 14
of 43. Per-trajectory reasons in `reachability_report.json`.

## Regenerating

```bash
export HOME=/home/huangsicheng
PY=$HOME/pro5000_env/.venv_isaacsim_pro5000/bin/python   # needs numpy
rm -rf tasks/libero/envs        # build_env_dirs.py refuses to overwrite
cd $HOME/benchmark && $PY bench_libero/tasks/libero/build_env/build_env_dirs.py
$PY bench_libero/tasks/libero/build_env/build_object_urdf.py   # the .obj + .urdf
$PY bench_libero/tasks/libero/build_env/verify_env_dirs.py     # expect FAILURES: 0
$PY bench_libero/tasks/libero/build_env/verify_object_urdf.py
$PY -m bench_libero tasks
```

`ENV_LIST.csv` + `flows/` + `pointflow/` are the inputs and are only ever read.

> **`rm -rf tasks/libero/envs` destroys `alphabet_soup/`'s 7 extra files.** They are
> the 0813 worked-example residue (usd/textures/config), and **no script here
> regenerates them** — `build_env_dirs.py` never writes usd or textures. If you want
> them, back that one directory up first. Everything the env *contract* needs is
> regenerated; only the residue is lost.

## Status (corrected 2026-09-06)

**The `authored_npz` path is wired.** This section previously said `driver.py` had
zero handling for it. That was written at 20:15 on 2026-09-01; the wiring landed at
20:52 the same evening and the text was never updated. What exists now:

| | |
|---|---|
| `driver.py:222` | reads `goal_source`, branches on `authored_npz` |
| `driver.py:158` | imports `build_authored_episode` from `envs/authored_episode.py` |
| `envs/authored_episode.py` | present, 15.6K |
| both libero suites | declare `donor_shard` (20260818 `derived_000427`) |

Commits: `a80df18 wire LIBERO object-flow end to end`, then
`9f65a61 22 LIBERO env dirs, spawnable URDFs, 95-case index`.

**Still true: no rollout has ever been run.** `verify_env_dirs.py` passing
(`FAILURES: 0`, `95 + 35 = 130`) is a *static* check — it does not start Isaac.
Run one case as an infrastructure gate before the full 95.

### Two constraints that are not going away

1. **The donor cube is 0.06 m; these objects are 0.077–0.354 m.**
   `authored_episode.py` synthesises a DerivedEpisode from a donor shard whose
   canonical cloud is a 0.06 m cube. That substitution was *exact* for authored cube
   paths. It is **not** exact here, and the error has never been measured. Measure it
   before reading any number — this is the next decision, not a detail.
   `chefmate_8_frypan` at 0.354 m is 5.9x the cube; its 7 tasks are a separate
   question, not part of a headline.
2. **LIBERO ships no robot joint data at all** — it is Franka + gripper, ours is
   xArm7 + Wuji 27 DoF. `envs/episode.py` requires `robot_joint_pos`/`_target`, so a
   teacher replay here is *structurally* impossible, not merely unimplemented.
   Consequence worth stating plainly: **the replay gate does not exist on LIBERO.**
   Per the 0827/0828 lessons that gate is the only thing that separates "the platform
   is broken" from "the policy is bad", so every LIBERO number must be read without it.

## Metrics change meaning

`guide_tracking_*` and `demo_success` compare against a **teacher recording**. No
teacher rollout exists behind a LIBERO trajectory, so the same formulas measure
agreement with a target path: "did it follow", not "did it reproduce". These numbers
must never be averaged in with held-out shard results.

## Known gap

`viewers/` holds **54 of 130** HTML files (930M). The background tar from js4 did not
finish; the light products (flows/pointflow/pipeline/samples) are complete at 130/130.
Stems present are listed by `ls viewers/`; the missing 76 need a re-transfer.
