# Evaluating your checkpoint

Your policy runs in **your** process, with your framework, your CUDA build, your
conda env. We never import it. You write one server script; we talk over pipes.

```
┌─ your env ─────────────┐          ┌─ Isaac env ─────────────┐
│ your_server.py         │   FIFO   │ TRO-MP sim + benchmark   │
│ + your checkpoint      │ <------> │ (physics gate, metrics)  │
└────────────────────────┘          └─────────────────────────┘
```

---

## The task

Each case starts from a **recorded initial state with the cube on the table** —
not pre-grasped. At reset you receive that episode's entire object-flow plan.

**You must grasp the cube yourself, then follow the plan.**

Both halves are scored. exp67 got `ever_lifted = 0/8`: it never finished the
first half, so the second was never tested.

---

## 1. Copy the reference server

```bash
cp ~/benchmark/scripts/policy_server_reference.py my_server.py
```

Edit exactly two things.

**`declare_contract()`** — what your policy is. Answered once, before the
simulator is even built, because the env branches on these fields instead of
inspecting your policy object. That indirection is what lets your checkpoint
live in another process.

```python
def declare_contract(args):
    return {
        "name": "my_policy_v3",
        "action_space": "joint",          # joint | ee_6d
        "action_dim": 27,                  # 7 arm + 20 hand
        "flow_frames": 768,
        "flow_conditioning": "full_plan",  # full_plan | per_step
        "flow_representation": "goal_residual",
    }
```

| field | meaning |
|---|---|
| `action_space` | `joint` → your 27 values go straight to PhysX. `ee_6d` → xyz + rot6d in the robot base frame, and **the env runs IK for you** |
| `action_dim` | length of the vector `act()` returns; the env raises on a mismatch |
| `flow_frames` | plan length you expect; cross-checked against the episode |
| `flow_conditioning` | `full_plan` → whole plan once at reset. `per_step` → one slice per tick inside the observation |
| `flow_representation` | `goal_residual` or `surface_point_tracks`; picks which shard array becomes the plan |

**`Policy`** — your checkpoint.

```python
class Policy:
    def __init__(self, checkpoint, action_dim, dummy):
        self.model = torch.load(checkpoint, map_location="cuda").eval()

    def reset(self, plan):
        # full_plan conditioning: this is the ONLY time you see the plan.
        self.plan = plan                   # [768, 1024, 3] float32

    def act(self, observation):
        return self.model(observation)      # -> (27,) float32
```

Observation keys, all float32, robot base frame:

```
robot_joint_pos        (27,)        live joint positions
object_surface_points  (1024, 3)    live object surface
object_flow            (...)        only when flow_conditioning == "per_step"
```

Check the protocol before involving Isaac at all:

```bash
python my_server.py --in-pipe /tmp/a --out-pipe /tmp/b --dummy
```

---

## 2. Run the evaluation

```python
from interprior_bench import run

report = run(
    policy="/your/conda/env/bin/python my_server.py --ckpt model.pt",
    suite="cube_lift_follow_v1",
    cases=8,          # how many episodes
    html_cases=2,     # how many 3-D replay pages
    gpus=[6, 7],      # one worker per GPU, cases split across them
)

print(report)
print(report.demo_success, report.guide_tracking_mean_m)
```

Or from the shell:

```bash
python -m interprior_bench.cli run \
  --policy "/your/env/bin/python my_server.py --ckpt model.pt" \
  --cases 8 --html-cases 2 --gpus 6,7
```

### Parameters worth knowing

| parameter | default | notes |
|---|---|---|
| `cases` | suite's own limit | how many episodes to evaluate |
| `html_cases` | `0` | 3-D replay pages. `-1` for all. **Each page is ~25 MB**, so this is opt-in |
| `html_frames` | `300` | playback frames per page |
| `frames` | `0` | `0` = full recording (2400). `None` = the 768 training window. `N` = first N |
| `gpus` | `[0]` | one subprocess each; Isaac Lab binds a process to one device at import |
| `resume` | `True` | a shard whose `summary.json` exists is skipped |

`gpus` gives crash isolation for free: one shard dying leaves the others'
results on disk, and re-running with `resume=True` picks up only what is missing.

---

## 3. Read the result

Primary metrics are **v10 guide tracking** — the same numbers every existing
`summary.json` reports, so your result sits beside exp67's without translation.

```
ever_lifted           did the policy get the cube off the table
dropped_after_lift    lifted, then lost it while the recording still held it
lift_and_hold         lifted, held ≥15 consecutive frames, still up at the end
lift_and_follow       lift_and_hold + tracked within 5 cm for ≥50% of frames
demo_success          ever_lifted, no drop, and trajectory_following
guide_tracking_mean_m mean distance from 1024 live surface points to recorded
```

`guide_tracking` uses **surface points, not the object centre** — that folds
orientation error into the metric. A cube rotated in place scores badly on
guide tracking and perfectly on centre distance.

> **Metric trap.** `guide_tracking_fraction_within_3cm ≈ 0.69` does *not* mean
> good tracking. It means the object never moved. Always read it next to
> `ever_lifted`.

The js4 four-class breakdown (`success` / `partial` / `drop` / `no_lift`) is
reported alongside for continuity with older runs.

### Output layout

```
<out>/
  report.json               merged across shards
  gpu6/summary.json         per-shard metrics + per-case detail
  gpu6/case0000_rollout.html    3-D replay: recorded path vs yours
  gpu6/replay_states.npz    every trajectory, for your own analysis
  gpu6/policy_server.log    your server's stdout/stderr
  logs/gpu6.log             worker log
```

In the 3-D pages, **cyan is the recorded path and orange is yours**. The gap
between the two moving markers is that frame's error.

---

## 4. Before trusting a number

Physics alignment is asserted **before** the simulator is built — 23 semantic
fields of the frozen data-production profile against the live config. A run
whose physics does not match the config the training data was produced under
raises instead of producing a plausible number.

Check it independently:

```bash
python -m interprior_bench.cli verify
```

Sanity-check the platform with teacher actions, which should reproduce the
recording almost exactly:

```bash
python -m interprior_bench.cli replay --cases 8 --html-cases 1
```

Current baseline on `cube_lift_follow_v1`, 8 cases × 2400 frames:

```
reference_success   8/8
guide mean          5.6 mm      (threshold 30 mm)
within 5 cm         100%
pos error median    2.25 mm
```

If your `demo_success` is 0 while replay passes, the platform is fine and the
finding is about your checkpoint.

---

## Wire protocol

You only need this if you are writing a server from scratch rather than copying
the reference. Framing is byte-identical to the existing ACT and flow servers.

```
header   >I    payload length in bytes
payload  <f4   float32 little-endian

contract    [-3.0]                  -> reply: UTF-8 JSON, same header form
reset       [-4.0]                  -> no reply
plan        [-5.0, F, P, D] ++ flat  -> reply: [1.0]
tick        [num_rows] ++ rows       -> reply: [num_rows * action_dim]
            row = [env_id, reset_flag, joint_pos(27), surface(1024*3), flow?]
shutdown    [-1.0]                  -> exit
```

Two traps, both of which have cost us time:

**Sentinel order.** `CONTRACT` (-3), `RESET` (-4) and `PLAN` (-5) are all frames
beginning with a negative value. A "negative means shutdown" test placed first
swallows the handshake. Dispatch named sentinels before the generic check.

**FIFO open order.** `open(O_WRONLY)` on a FIFO blocks until a reader attaches.
The driver opens `--in-pipe` for writing first, then `--out-pipe` for reading;
mirror that order in your server or both sides wait forever. The adapter polls
with `O_NONBLOCK` and reaps the child, so a server that fails to start raises
with its own log rather than hanging — but only on our side.

---

## Environment

Isaac needs an NVIDIA node. `bench tasks` and the protocol smoke tests run
anywhere.

```bash
ssh -p 10670 huangsicheng@42.192.34.154        # Pro5000, 8 GPUs
export TMPDIR=$HOME/tmp; export OMNI_KIT_ACCEPT_EULA=YES
```

`/home` and `/mnt` are shared across nodes, so nothing needs copying. See
`log/gpu_use.md` for which nodes have Isaac and which are busy.
