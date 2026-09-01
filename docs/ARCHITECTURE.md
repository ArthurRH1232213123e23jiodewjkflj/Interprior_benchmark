# interprior_bench

Evaluate a manipulation policy on TRO-MP cube lift-and-follow, in Isaac Lab,
against the physics the training data was actually produced under.

Your checkpoint stays in your own process, your own conda env, your own
framework. The benchmark never imports it.

```python
from interprior_bench import run

report = run(
    policy="/your/env/bin/python my_server.py --ckpt model.pt",
    cases=8,          # how many episodes
    html_cases=2,     # how many 3-D replay pages
    gpus=[6, 7],
)
print(report)
```

Full integration guide: **[docs/API.md](docs/API.md)**.

---

## The task

Each case starts from a recorded initial state with the cube **on the table** —
not pre-grasped (`start_frame=0`). At reset the policy receives that episode's
entire object-flow plan.

**Grasp the cube, then follow the plan.** Both halves are scored.

---

## Three commands

Run these on a Pro5000 node (see [Nodes](#nodes)).

```bash
python -m interprior_bench.cli tasks     # what suites exist
python -m interprior_bench.cli verify    # is the live physics the real physics
python -m interprior_bench.cli replay    # teacher actions: does the platform work
python -m interprior_bench.cli run --policy "..."   # evaluate a checkpoint
```

`verify` and `replay` are the two gates. Run them before trusting any policy
number — if replay passes and your policy scores zero, the finding is about your
checkpoint, not the platform.

---

## How a policy connects

Three paths, same worker and same metrics.

| path | flag | when |
|---|---|---|
| **subprocess IPC** | `--policy-cmd "..."` | your checkpoint, any framework, its own env. **This is the one for external users.** |
| in-process | `--policy-ckpt` + `--policy-config` | flow_policy checkpoints, loaded directly. No pipe, so a wrong number can't be blamed on the wire. |
| recorded teacher | *(default)* | replays the shard's own actions. This is the physics gate, not an evaluation. |

For IPC, copy `scripts/policy_server_reference.py` and edit two functions:

```python
def declare_contract(args):        # what you are
    return {"action_space": "joint", "action_dim": 27,
            "flow_frames": 768, "flow_conditioning": "full_plan", ...}

class Policy:
    def reset(self, plan):         # full_plan: the ONLY time you see it
        self.plan = plan
    def act(self, observation):    # joint_pos(27) + surface_points(1024,3)
        return self.model(observation)     # -> (27,)
```

`--dummy` runs the protocol without a checkpoint, so you can test the wiring
first. Two smoke tests need no GPU at all:

```bash
python scripts/ipc_smoke.py     # handshake, round trip, shutdown, failure surfaces
python scripts/ipc_smoke2.py    # batched ticks + 9.4 MB plan delivery
```

### Why a contract instead of a policy object

The upstream loop read ~10 attributes straight off an in-process runner
(`runner.uses_ee_arm`, `runner.flow_frames`, …) and changed its own behaviour
accordingly — which is exactly why a foreign checkpoint could not be evaluated.
Those questions are now fields the adapter **declares once at handshake**, so the
env inspects nothing but a dataclass.

---

## Metrics

Primary set is v10 guide tracking — the same numbers every existing
`summary.json` reports, so a result sits beside exp67's without translation.

```
ever_lifted            did the policy get the cube off the table
dropped_after_lift     lifted, then lost it while the recording still held it
lift_and_hold          lifted, held >=15 consecutive frames, still up at the end
lift_and_follow        lift_and_hold + within 5 cm for >=50% of frames
demo_success           ever_lifted, no drop, and trajectory_following
guide_tracking_mean_m  mean distance, 1024 live surface points to recorded
```

Guide tracking uses **surface points, not the object centre**, so orientation
error counts. A cube rotated in place scores badly here and perfectly on centre
distance.

> **Trap.** `guide_tracking_fraction_within_3cm ≈ 0.69` does not mean good
> tracking — it means the object never moved. Always read it next to
> `ever_lifted`.

The js4 four-class breakdown (`success` / `partial` / `drop` / `no_lift`) is
reported alongside for continuity with earlier runs.

---

## Physics truth

Alignment is asserted **before** the simulator is built: 23 semantic fields of
the frozen data-production profile against the live config. Mismatch raises
rather than producing a plausible number.

```
profile   interprior_bench/physics/profiles/tro_mp_dataprod_venvshare_20260808.yaml
sha256    e9de8133c675f2d4c1eb69be09d9a143eadfddb637bf57102b9d68a66e6c0935
source    /mnt/venv_share/{pro5000,H800}/Interprior   (byte-identical)
```

This is the config the 20260818 shards were produced under, confirmed three ways:
the collection logs, zjw's 8-24 eval, and v10's own loader template.

**It is not the config the v10 eval path pins.** That one (`ce6dfb05`) differs by
160 lines, including velocity solver iterations 64/8 → 0/0 and a round table that
silently reverts to a square one because the key is simply absent. The gate
catches all of it — 12 fields — and `_wrong_v10eval_pairing_ce6dfb05.yaml` is
kept beside the real profile as the counterexample.

Why semantic assertions and not a digest: the 20260818 shards record no
`task_config_sha256` at all, so upstream's single-file hash check is a no-op on
this data. Details in `log/physics_difference.txt`.

---

## Verified

Teacher replay, 8 cases × 2400 frames, on the data-production physics:

```
reference_success   8/8
guide mean          5.6 mm      (threshold 30 mm)
within 5 cm         100%
pos error median    2.25 mm     (physics gate: <= 3 mm)
four-class          success 8
```

exp66@44000 through the full policy path (in-process, same suite, 768 frames):

```
ever_lifted    0/8
guide mean     51.0 mm
four-class     no_lift 8
```

That reproduces exp67's recorded `ever_lifted 0/8` under *different* physics,
which says the failure is policy-side, not a physics artifact.

**A caveat on that run:** `pos_err_median` and its gate are meaningful only for
teacher replay. In policy eval the object barely moves, so the median lands on
the float32 floor and the gate reads PASS regardless. Read the v10 metrics
instead. (Known issue: the gate line should be suppressed in policy mode.)

---

## Layout

```
interprior_bench/
  api.py              run() + multi-GPU orchestration, shard merge, resume
  cli.py              bench tasks | verify | replay | run
  protocol.py         wire format: >I header, <f4 payload
  envs/
    contract.py       PolicyContract — the handshake
    tro_mp.py         the env, ported from sim_rollout_render.py
    episode.py        v10 shard loader (zarr + parquet)
    rotation.py       rot6d / quaternion helpers
  adapters/
    base.py           PolicyAdapter ABC
    subprocess_adapter.py   IPC — the path for foreign checkpoints
    flow_policy_v10.py      reference adapter, in-process
    recorded_teacher.py     the physics gate
  physics/
    profiles/*.yaml   frozen truth + the known-wrong pairing
    verify.py         23-field assertion
  suites/             declarative tasks; adding one touches no code
  render/
    rollout_viewer.py Three.js pages: recorded path vs replayed
    flow_viewer.py    page builder (numpy only)
  report/
scripts/
  bench_replay.py     the worker: gate, rollout, metrics, pages
  verify_physics_live.py
  policy_server_reference.py   copy this
  ipc_smoke.py  ipc_smoke2.py  protocol tests, no GPU needed
docs/API.md
_upstream/            verbatim copies of what was ported, for audit
```

Output of a run:

```
<out>/report.json                     merged across shards
     gpu6/summary.json                per-shard metrics + per-case detail
     gpu6/case0000_rollout.html       3-D replay, cyan = recorded, orange = yours
     gpu6/replay_states.npz           every trajectory
     gpu6/policy_server.log           your server's output
     logs/gpu6.log
```

---

## Nodes

Isaac needs NVIDIA. `bench tasks` and the smoke tests run anywhere.

| node | port | Isaac | notes |
|---|---|---|---|
| Pro5000_08 | 10670 | yes | 8 GPUs idle — **use this** |
| Pro5000_07 | 11148 | yes | mostly idle |
| Pro5000_05 | 11002 | yes | busy, 6/8 GPUs at 60–99% |
| hy_02 | 11009 | **no** | Hygon DCU, no NVIDIA. Write code here, run it there. |

`/home` and `/mnt` are shared, so nothing needs copying between nodes.
Full survey in `log/gpu_use.md`.

```bash
ssh -p 10670 -i ~/.ssh/id_ed25519 huangsicheng@42.192.34.154
export TMPDIR=$HOME/tmp; export OMNI_KIT_ACCEPT_EULA=YES
```

### Isaac python

The benchmark resolves it in order, and **does not fall back to anyone else's
venv** — a path in another user's home is a symlink they can repoint, which would
silently change the Isaac build under a validated result.

```
$INTERPRIOR_BENCH_ISAAC_PYTHON
~/pro5000_env/.venv_isaacsim_pro5000/bin/python      <- ours
~/Interprior_bench_env/.venv_isaacsim_pro5000/bin/python
~/code/Interprior/.venv_isaacsim_pro5000/bin/python
```

None found raises with the build command. To build one (`uv` is only on the
Pro5000 nodes, so build there):

```bash
cp -a /mnt/venv_share/pro5000/Interprior ~/pro5000_env
rm -rf ~/pro5000_env/.venv_isaacsim        # the script symlinks the shared tree

CODE_DIR=~/pro5000_env \
  bash /mnt/venv_share/pro5000/build_pro5000_venv_isaacsim.sh
```

Takes ~100 s. `CODE_DIR` must be an Interprior checkout: the build script writes
it into a `.pth`, so it *is* the source of `isaacsimenvs` — the other half of
physics truth, which is why the copy comes from the data-production checkout.
Verify after copying:

```bash
sha256sum ~/pro5000_env/isaacsimenvs/cfg/task/TroMp.yaml
# e9de8133c675f2d4c1eb69be09d9a143eadfddb637bf57102b9d68a66e6c0935
```

---

## Not done yet

- `--policy-cmd` (IPC) is verified at the protocol level by both smoke tests, but
  no real checkpoint has gone through it end to end — only the in-process path has.
- `TroMpEnv` is unused: `bench_replay.py` builds its own env, so there are two
  env paths and the validated one is not the class. Should converge before more
  is built on either.
- `pos_err` gate leaks into policy-eval output (see the caveat above).
- One suite. `cube_lift_v1` and `cube_goal_authored_v1` are planned; the
  goal_traj toolchain for the latter is already vendored under `tasks/goal_traj/`.

---

## History

This README describes what the benchmark *is*. How it got here — the physics
investigation, the port decisions, every bug and its root cause — is in:

```
~/log/0825_benchmark.md        build log, decisions, results
~/log/physics_difference.txt   the four TroMp.yaml files and why it matters
~/log/gpu_use.md               node survey, venv per architecture
MANIFEST.md                    what was ported from where, and what was not
```
