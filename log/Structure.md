# Working notes — interprior_bench

Live state, conventions, and the two pipelines that matter. `README.md` says what
the benchmark *is*; this file says how to work on it and what is currently true.

> There is also a `progess.md` (one `r`) at this root — a stale duplicate of
> `docs/ARCHITECTURE.md`. Ignore it. This file is the one to update.

---

## Rule 0 — extend the existing code, do not rebuild it

Every capability below already exists. The failure mode on this project is not
missing features, it is a second implementation of a feature that was already
there, with slightly different behaviour. Two of those have already cost real
time, both recorded in `docs/ARCHITECTURE.md` under "Not done yet".

**Before writing anything:**

1. Read `README.md` (integration), `docs/API.md` (contract + wire), and
   `docs/ARCHITECTURE.md` (layout, what is unfinished).
2. Search for it first. `grep -rn` the name of the thing you are about to write.
3. Add a suite or a profile, not a code path. Suites and physics profiles are
   *declarative on purpose* — a new dataset or a new task should touch **zero**
   Python.
4. If you must change shared code under `interprior_bench/`, keep a `.bak_*`
   beside it and re-run the replay gate (below) before trusting anything.
5. Your own work goes in `train_eval/`. Vendor what you depend on; never load a
   checkpoint or a library out of another user's live repo at run time — they can
   change it under a validated result.

Concrete instances of rule 3, both real:

- exp04 needed a different dataset, a different plan length, and different
  physics. That was **one suite yaml + one profile yaml**, no new loader.
- The 20260821 shards did not load. The fix was widening one dispatch condition
  in the existing reader, **not** a second reader — the reader was already
  correct, `training_full` is a superset of `training_min`.

---

## Current state (2026-08-27)

| piece | state |
|---|---|
| teacher replay on 20260818 | works — the published baseline |
| in-process policy path | works (exp66@44000 measured) |
| **subprocess IPC path** | server verified end to end on CPU with a real ckpt; **no完整 GPU rollout yet** |
| 20260821 `training_full` loading | **fixed** — dispatch widened |
| 20260821 physics profile | **frozen** — `tro_mp_lqr_20260821.yaml` |
| exp04 on 200 held-out cases | **not yet run** |

### exp04 evaluation — what is built

Under `train_eval/`:

```
policy_server_exp04.py    wraps FlowPolicyRunner behind the IPC protocol
ckpt/                     exp04 ckpt + model config, copied; sha in PROVENANCE.txt
vendor/flow_policy/       frozen copy — nothing loads from a live repo
suites/exp04_test200_v1.yaml     the suite
suites/exp04_test200_cases.json  200 held-out cases
registry.json             all three datasets, each pointing at its protocol doc
freeze_profile.py         make a physics profile for a new dataset
build_test200_cases.py    selection.json -> case index
```

Facts worth not rediscovering:

- **Only `ckpt_0055000.pt` is readable to us.** Every other exp04 checkpoint is
  `-rw-------`.
- exp04 trained with `overfit_all_data=true` and logged **no validation**, so a
  held-out split is the only source of a generalisation number for it.
- The 200 cases are zjw's own `test_selection.json`
  (`seed 768500200`, sha `7b1fe52b…`), **overlap 0** with the 500 it trained on,
  verified by `episode_uid`. Do not re-select them; a different set is a
  different number.
- Contract read off the checkpoint: 27-dim joint action, `full_plan`,
  `surface_point_tracks`, plan 647 frames, 1024 source points → 64 internally
  (`evenly_spaced`).
- `runner.flow_frames` is **0 before reset** (it reads the installed plan's
  shape). Declare the length from `object_flow_plan_max_frames` instead, or the
  env cross-checks against nothing. Same quirk the reference adapter documents.
- `SubprocessAdapter.commit_action()` is a **no-op** — the base method. The
  driver calls it after its own clipping, so nothing crosses the wire. Chunking
  policies must commit locally in the server or they desynchronise.
- `start_frame: 0` is correct even though exp04 cropped its plan to source frames
  `[121, 768)`: frames 0–121 are **bit-identical** to 121 (arm/hand max |Δ|
  0.0000 rad, object 0.00 mm, measured on three episodes). That prefix is the
  static settle. The model never sees an absolute frame index.

---

## Physics profiles — and adding a new dataset's physics

### Why 20260821 needed its own

The gate asserts the live sim against a frozen profile. exp04 trained on
**20260821 LQR** data, but the only profile was frozen from **20260818**. They
differ:

| field | 0818 | 0821 |
|---|---|---|
| `reset.object_reset_annulus_r_in_m` | 0.30 | **0.32** |
| `reset.object_reset_annulus_r_out_m` | 0.60 | **0.55** |
| `reset.object_reset_z_offset_m` | 0.03 | **0.0** |
| `reset.object_reset_orientation_mode` | `random` | **`yaw_only`** |
| `tro_mp.grasp_buffer_path` | `idx0_520/` | **`idx0_64/`** |
| `tro_mp.output_mode` | `episode_hdf5_async` | **`shard`** |

Scoring 0821 episodes against the 0818 profile measures that gap and calls it
policy performance. Hence `tro_mp_lqr_20260821.yaml`, frozen from 0821's own
`task_config_resolved.yaml` (sha `ce0a6f7f…`).

Two things this comparison settled, both worth keeping:

- The fields that actually act **during** a rollout — `squeeze_fraction` 0.5,
  `object_flow_num_points` 1024 — already matched the live checkout. The four
  `reset.*` differences only govern how the cube is randomly placed at episode
  start, and this eval restores the recorded initial state from the shard
  (`start_frame=0`), so they never execute. They are still pinned honestly to the
  0821 values rather than waved through.
- **zjw's eval checkout (`/home/zhangjiawei/Interprior`) is not a substitute.**
  It declares `squeeze_fraction: 0.6` and `object_flow_num_points: 16` — both act
  during a rollout, so borrowing it silently changes the grasp physics. Use
  `/mnt/venv_share/H800/Interprior`, which matches 0821 on every pinned field.

### A profile has TWO jobs — this is the trap

1. `physics/verify.py` compares ~23 semantic fields of the live cfg against it.
2. `scripts/bench_replay.py` **overlays the entire profile onto cfg** before
   building the sim.

A profile trimmed to only the verified fields satisfies (1) and breaks (2). It
fails as:

```
ValueError: cfg.assets.object_scale must be set when object_urdf is given
```

`assets.object_scale` is overlay-only — never checked by the gate, required by the
env. Roughly 80 keys are in that category. **So a new profile must keep an
existing profile's full key set** and change only the values.

### Adding physics for a new dataset

```bash
cd ~/benchmark
python train_eval/freeze_profile.py \
  --template interprior_bench/physics/profiles/tro_mp_dataprod_venvshare_20260808.yaml \
  --resolved  /path/to/<dataset>/.../task_config_resolved.yaml \
  --out       interprior_bench/physics/profiles/tro_mp_<name>.yaml \
  --label     "<human label>" \
  --dataset   <registry key> \
  --force     scene.num_envs=1
```

`--template` supplies **which** keys to declare; `--resolved` supplies the
**values**. It prints every verified field that moved, so the physics delta is
reviewable instead of implicit. `--force scene.num_envs=1` is normal: collection
runs thousands of parallel envs, evaluation runs one — that is scale, not physics.

Then point a suite at it:

```yaml
physics_profile: tro_mp_<name>.yaml
interprior_root: /mnt/venv_share/H800/Interprior
```

Finally register the dataset in `train_eval/registry.json` — each entry records
schema, profile, frame conventions, counts, and a link to zuoyufan's own protocol
doc, which is authoritative when the two disagree.

Two rules both protocol docs state and the registry exists to enforce:

- **Never glob a dataset root to discover samples.** Enumerate from a frozen
  selection or manifest, or the sample set grows silently under a result.
- **`episode_status` ≠ `training_eligible`.** Filtering on `success` is not the
  same as having a usable flow.

### Datasets currently declared

| key | data | profile | state |
|---|---|---|---|
| `tro_mp_cube_20260818` | teacher cube, `training_min` | `…venvshare_20260808` | baseline |
| `lqr_cube_4096env_20260821` | LQR cube, `training_full` | `tro_mp_lqr_20260821` | ready |
| `tro_objaverse_1m` | Objaverse multi-object | — | **not wired**: unfrozen physics, uses `episodes.parquet`, needs loader work |

### The loader change that unblocked 0821

`envs/episode.py` dispatched the parquet reader only on
`profile == "training_min"`, so `training_full` shards fell through to a branch
hardcoding `episode_table.jsonl` — which those shards do not have. Widened to
accept both, because `training_full` is a **superset**: same eight parquet
columns, every raw frame array, plus `_w` world-frame copies.

Verified before relying on it: `mixed_world_and_robot_base` does **not** mean the
unsuffixed arrays became world-frame. For one 0821 episode the unsuffixed pose was
`[0.311, -0.172, 0.030]` (base) against `[-1.489, -23.573, 0.560]` (world). The
reader touches only unsuffixed arrays, so it is correct for either profile. That
check was the point — feeding world-frame poses to a base-frame policy yields a
plausible, wrong score rather than an error, so `OBJFLOWNET_V10_BASE_FRAMES`
gates it explicitly.

Backup: `envs/episode.py.bak_pre_trainingfull`.

---

## Replay pipeline — the gate, and how to read it

**Replay is not an evaluation.** It feeds the shard's own recorded teacher actions
back through the simulator and asks whether the platform reproduces the recording.
It is the one test that separates "the platform is broken" from "the policy is
bad", so it runs *before* any policy number is trusted.

```bash
export INTERPRIOR_BENCH_ISAAC_PYTHON=$HOME/pro5000_env/.venv_isaacsim_pro5000/bin/python
export TMPDIR=$HOME/tmp
export OMNI_KIT_ACCEPT_EULA=YES

cd ~/benchmark
python -m interprior_bench.cli replay --cases 8 --gpus 6 --html-cases 1
```

Baseline on `cube_lift_follow_v1` (20260818, 8 cases × 2400 frames):

```
reference_success   8/8
guide mean          5.6 mm      (threshold 30 mm)
within 5 cm         100%
pos error median    2.25 mm     (physics gate: <= 3 mm)
```

### What it does, in order

1. **Physics gate.** Asserts ~23 semantic fields of the live cfg against the
   suite's frozen profile — *before* the simulator is built, so a mismatch raises
   instead of producing a plausible number. Prints
   `physics gate: ok=True checked=N mismatch=0`.
2. **Load episodes** from the derived shards named by the suite's case list.
3. **Restore** each case's recorded initial state (`start_frame`).
4. **Step**, injecting recorded `robot_joint_pos_target` each tick — no policy,
   no clipping (recorded targets are ground truth).
5. **Score** against the recording: guide tracking over 1024 surface points, plus
   `pos_err_median` as the physics gate.
6. **Render** optional 3-D pages — cyan is recorded, orange is replayed.

### Interpreting it

- **replay passes, policy scores 0** → platform is fine, the finding is about the
  checkpoint. This is the whole reason replay exists.
- **replay fails** → stop. Do not read the policy number at all. Suspect the
  profile/checkout pairing first.
- `pos_err_median` and its PASS/FAIL line are **meaningful only for replay**. In
  policy eval the object barely moves, the median lands on the float32 floor, and
  the gate reads PASS regardless. Known issue; read the v10 metrics instead.
- `guide_tracking_fraction_within_3cm ≈ 0.69` does **not** mean good tracking — it
  means the object never moved. Always read it next to `ever_lifted`.

### Replay on a new dataset

Same command, different suite: replay uses the `recorded_teacher` adapter
automatically when no `--policy` is given, so a suite pointed at new shards is
immediately replayable. Do this **first** on any new dataset — it validates the
loader, the profile, and the case list in one shot, without a policy in the way.

```bash
python -m interprior_bench.cli replay --suite train_eval/suites/<new>.yaml --cases 8 --gpus 6
```

### Protocol tests — no GPU needed

Separate "my wire format is wrong" from "my model is wrong" before spending GPU
time:

```bash
python scripts/ipc_smoke.py                      # handshake, round trip, shutdown
python scripts/ipc_smoke2.py                     # batched ticks + 9.4 MB plan
python train_eval/smoke_exp04_server.py          # our server, dummy
python train_eval/smoke_exp04_server.py --real   # our server, real ckpt on CPU
```

---

## Policy evaluation

```bash
export INTERPRIOR_BENCH_ISAAC_PYTHON=$HOME/pro5000_env/.venv_isaacsim_pro5000/bin/python
export TMPDIR=$HOME/tmp; export OMNI_KIT_ACCEPT_EULA=YES

PY=$HOME/.conda/envs/interprior/bin/python
cd ~/benchmark
$PY -m interprior_bench.cli run \
  --policy "$PY train_eval/policy_server_exp04.py --strict-finite" \
  --suite  train_eval/suites/exp04_test200_v1.yaml \
  --cases 8 --gpus 6 \
  --out    train_eval/eval/exp04_test200
```

Two pythons, and confusing them wastes an afternoon:

- **Isaac python** (`INTERPRIOR_BENCH_ISAAC_PYTHON`) runs the simulator side.
- **policy python** (inside `--policy`) runs your checkpoint, in its own env. The
  pipe exists precisely so these two need not be compatible.

`--gpus` takes **physical** indices from `nvidia-smi`, unvalidated — check first,
the box is shared. One worker per GPU; cases are split round-robin, and a dead
shard leaves the others' results on disk for `resume`.

### Open item

`--interprior-root` defaults in the CLI **override** the suite's own
`interprior_root`, which produces:

```
WARN interprior_root=/home/huangsicheng/pro5000_env is not one of the
data-production checkouts (...); env code and yaml may be mismatched
```

The gate still passes (the yaml matches), but env *code* comes from the wrong
checkout. Either pass `--interprior-root /mnt/venv_share/H800/Interprior`
explicitly, or make the suite field win. Worth fixing properly — it is exactly
the class of silent mismatch the gate exists to catch.

---

## Nodes

`/home` and `/mnt` are shared CFS across all nodes, so nothing needs copying —
write code on one, run on another, same paths.

| node | port | Isaac | note |
|---|---|---|---|
| Pro5000_05 | 11002 | yes | 8 GPUs idle as of 2026-08-27 (docs saying "busy" are stale) |
| Pro5000_08 | 10670 | yes | partly busy |
| Pro5000_07 | 11148 | yes | busy |
| hy_02 | 11009 | **no** | Hygon DCU, no NVIDIA. Write here, run there. |

On 11009 every command needs `export HOME=/home/huangsicheng` first —
`/etc/environment` forces `HOME=/root`, which is unwritable.

Isaac on a Hygon node fails **15 s after Kit starts, with exit code 0**, which
reads like a corrupt checkpoint. `require_nvidia_node()` now fails fast instead.

---

## History

- `~/log/0825_benchmark.md` — build log, decisions, results
- `~/log/physics_difference.txt` — the four `TroMp.yaml` files and why it matters
- `~/log/gpu_use.md` — node survey, venv per GPU architecture
- `MANIFEST.md` — what was ported from where, and what deliberately was not
