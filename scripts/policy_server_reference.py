#!/usr/bin/env python
"""Reference policy server — copy this file and replace two functions.

This is the whole integration surface. Your checkpoint stays in your own conda
env with your own framework and your own CUDA build; the benchmark never imports
it. We talk over two FIFOs using length-prefixed float32 frames.

    ┌─ your env ────────────┐         ┌─ Isaac env ──────────────┐
    │ this script + your     │  FIFO   │ TroMp sim + benchmark    │
    │ checkpoint             │ <-----> │ (subprocess_adapter)     │
    └───────────────────────┘         └──────────────────────────┘

To adapt it, edit exactly two things:

    declare_contract()   what your policy is (action space, flow conditioning)
    load_policy() / Policy.act()   your checkpoint and its forward pass

Run it yourself to check the protocol before involving Isaac at all:

    python policy_server_reference.py --in-pipe /tmp/a --out-pipe /tmp/b --dummy

The benchmark normally spawns it for you, appending --in-pipe/--out-pipe:

    from bench_cube_val.adapters.subprocess_adapter import SubprocessAdapter
    adapter = SubprocessAdapter(server_cmd=[
        "/your/conda/env/bin/python", "policy_server_reference.py",
        "--ckpt", "model.pt",
    ])

WIRE PROTOCOL (framing is byte-identical to the ACT/flow servers in
_upstream/servers_ref/, so anything that spoke to those speaks to this):

    header   >I   payload byte length
    payload  <f4  float32 little-endian

    contract request   [-3.0]                 -> reply: UTF-8 JSON, same header
    reset              [-4.0]                 -> no reply
    plan               [-5.0, F, P, D] ++ flat -> reply: [1.0]
    per tick           [num_rows] ++ rows     -> reply: [num_rows * action_dim]
                       row = [env_id, reset_flag, observation...]
    shutdown           [-1.0]                 -> exit

One trap worth knowing: the contract sentinel (-3.0) is also a size-1 negative
frame, so a naive "size 1 and negative means shutdown" test eats the handshake.
Check for the contract sentinel first — this file does.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from typing import Any, BinaryIO

import numpy as np

# --- sentinels (must match bench_cube_val/protocol.py) --------------------
CONTRACT = -3.0
SHUTDOWN = -1.0
HANDSHAKE = -2.0  # legacy points_obj handshake, unused here
RESET = -4.0      # new episode, no plan attached
PLAN = -5.0       # [-5, frames, points, dims, env_id] ++ flat plan; reply [1.0]
COMMIT = -6.0     # [-6, num_rows, tol] ++ executed actions; reply [1.0]
INIT_STATE = -7.0 # [-7, dim, env_id] ++ recorded initial joints; reply [1.0]


# --- framing ---------------------------------------------------------------
def read_msg(f: BinaryIO) -> np.ndarray | None:
    """One length-prefixed float32 frame. None on clean EOF."""

    header = f.read(4)
    if len(header) < 4:
        return None
    size = struct.unpack(">I", header)[0]
    if size == 0:
        return np.empty(0, dtype=np.float32)
    buf = b""
    while len(buf) < size:
        chunk = f.read(size - len(buf))
        if not chunk:
            return None
        buf += chunk
    return np.frombuffer(buf, dtype="<f4")


def write_msg(f: BinaryIO, values: Any) -> None:
    array = np.asarray(values, dtype="<f4")
    f.write(struct.pack(">I", array.nbytes))
    f.write(array.tobytes())
    f.flush()


def write_json(f: BinaryIO, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    f.write(struct.pack(">I", len(body)))
    f.write(body)
    f.flush()


# --- EDIT ME 1: declare what your policy is --------------------------------
def declare_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Answered once, before the simulator is even built.

    The env branches on these fields instead of inspecting your policy object,
    which is what lets your checkpoint live in another process entirely.

    action_space         "joint"  -> action is 27 joint targets, straight to PhysX
                         "ee_6d"  -> action is xyz + rot6d in the robot base
                                     frame; the ENV runs IK, you do not
    action_dim           length of the vector your act() returns
    flow_frames          plan length you expect; the env cross-checks and raises
    flow_conditioning    "full_plan" -> whole immutable plan handed over at reset
                         "per_step"  -> one slice per tick inside the observation
                         "teacher_action_not_applicable" -> you consume no flow
    flow_representation  "goal_residual" | "surface_point_tracks"
                         picks which shard array becomes the plan

    Optional evaluation gates (omit or null to skip): dataset_kind,
    dataset_mode, plan_exact_frames, required_selection_split,
    static_arm_target_threshold_rad, episode_selection_sha256.
    """

    return {
        "name": args.name,
        "action_space": "joint",
        "action_dim": 27,
        "flow_frames": args.flow_frames,
        "flow_conditioning": "full_plan",
        "flow_representation": "goal_residual",
    }


# --- EDIT ME 2: your checkpoint --------------------------------------------
class Policy:
    """Wrap your model. Only `act` is called per tick."""

    def __init__(self, checkpoint: str | None, action_dim: int, dummy: bool) -> None:
        self.action_dim = action_dim
        self.dummy = dummy
        self.model = None
        self.plan = None
        if dummy or checkpoint is None:
            print("[server] dummy mode: emitting the observed joint position",
                  file=sys.stderr, flush=True)
            return
        # --- replace with your own loading ---------------------------------
        # import torch
        # self.model = torch.load(checkpoint, map_location="cuda").eval()
        raise SystemExit(
            f"replace Policy.__init__ to load {checkpoint}, or pass --dummy"
        )

    def reset(self, plan: np.ndarray | None) -> None:
        """Called when a fresh episode starts.

        `plan` is the immutable object-flow plan, [frames, points, 3] for
        surface_point_tracks or [frames, points, 3] residuals for goal_residual —
        whichever your contract declared. For full_plan conditioning this is the
        ONLY time you see it: the task is to grasp the cube and then follow it.
        """

        self.plan = plan
        if plan is not None:
            print(f"[server] plan received: {plan.shape}", file=sys.stderr, flush=True)

    def act(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Return one action vector of length action_dim.

        observation keys, all float32 in the robot base frame:
          robot_joint_pos        (27,)      live joint positions
          object_surface_points  (P, 3)     live object surface, P = 1024
          object_flow            (...)      present only for per_step conditioning
        """

        if self.dummy:
            # Hold position: echo the joints back as the target. Physically inert,
            # which is exactly what a protocol test wants.
            joints = observation.get("robot_joint_pos")
            if joints is not None and joints.size == self.action_dim:
                return joints.astype(np.float32)
            return np.zeros(self.action_dim, dtype=np.float32)
        # --- replace with your own forward pass -----------------------------
        raise NotImplementedError


# --- observation unpacking -------------------------------------------------
def unpack_observation(
    row: np.ndarray,
    *,
    joint_dim: int,
    num_points: int,
    expect_flow: bool,
) -> tuple[int, bool, dict[str, np.ndarray]]:
    """Split one wire row into (env_id, reset_flag, observation).

    Layout, in the order subprocess_adapter writes it:
        [env_id, reset_flag, robot_joint_pos(joint_dim),
         object_surface_points(num_points * 3), object_flow(rest)?]

    Widths are not on the wire, so both sides must agree on joint_dim and
    num_points. They come from the task, not the policy: TRO-MP is 27 and 1024.
    Whether object_flow follows is decided by your own flow_conditioning —
    full_plan means it does not.
    """

    env_id = int(row[0])
    reset_flag = bool(row[1])
    cursor = 2

    joints = row[cursor:cursor + joint_dim].astype(np.float32)
    cursor += joint_dim

    surface_size = num_points * 3
    surface = row[cursor:cursor + surface_size].astype(np.float32)
    cursor += surface_size
    if surface.size != surface_size:
        raise ValueError(
            f"expected {surface_size} surface values, got {surface.size}; "
            "joint_dim/num_points disagree with the driver"
        )

    observation = {
        "robot_joint_pos": joints,
        "object_surface_points": surface.reshape(num_points, 3),
    }
    if expect_flow and cursor < row.size:
        observation["object_flow"] = row[cursor:].astype(np.float32)
    return env_id, reset_flag, observation


def open_pipes(in_pipe: str, out_pipe: str):
    """Open both FIFOs without deadlocking against the driver.

    A blocking O_RDONLY open on a FIFO does not count as an open read end while
    it waits, so the driver's O_WRONLY|O_NONBLOCK keeps failing with ENXIO and
    neither side ever connects. Opening non-blocking registers the reader at
    once; O_NONBLOCK is then cleared so reads block normally.
    """

    import fcntl
    import os

    handle = os.open(in_pipe, os.O_RDONLY | os.O_NONBLOCK)
    flags = fcntl.fcntl(handle, fcntl.F_GETFL)
    fcntl.fcntl(handle, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
    fin = os.fdopen(handle, "rb")
    # The write end may block until the driver opens its read end, which it does
    # immediately after our read end appears -- so this resolves on its own.
    fout = open(out_pipe, "wb")
    return fin, fout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-pipe", required=True, help="env -> policy FIFO")
    parser.add_argument("--out-pipe", required=True, help="policy -> env FIFO")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--name", default="reference_server")
    parser.add_argument("--flow-frames", type=int, default=768)
    parser.add_argument("--joint-dim", type=int, default=27)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--dummy", action="store_true",
                        help="skip the checkpoint and hold position; use this to "
                             "test the protocol on its own")
    args = parser.parse_args()

    contract = declare_contract(args)
    policy = Policy(args.ckpt, int(contract["action_dim"]), args.dummy)
    expect_flow = contract["flow_conditioning"] == "per_step"
    action_dim = int(contract["action_dim"])

    # See open_pipes: opening the read end non-blocking is what lets the
    # driver's O_WRONLY succeed instead of spinning on ENXIO.
    fin, fout = open_pipes(args.in_pipe, args.out_pipe)
    with fin, fout:
        print(f"[server] connected: {args.in_pipe} -> {args.out_pipe}",
              file=sys.stderr, flush=True)
        ticks = 0
        while True:
            request = read_msg(fin)
            if request is None:
                print("[server] driver closed the pipe", file=sys.stderr, flush=True)
                break
            if request.size == 0:
                continue

            # Named sentinels FIRST. CONTRACT/RESET/PLAN all start with a
            # negative value, so the generic "negative means shutdown" test has
            # to come last or it swallows them.
            head = float(request[0])
            if request.size == 1 and head == CONTRACT:
                write_json(fout, contract)
                print(f"[server] contract sent: {contract['action_space']} "
                      f"dim={action_dim} flow={contract['flow_conditioning']}",
                      file=sys.stderr, flush=True)
                continue
            if request.size == 1 and head == RESET:
                policy.reset(None)
                print("[server] reset (no plan)", file=sys.stderr, flush=True)
                continue
            if head == INIT_STATE and request.size >= 2:
                # Recorded initial joints. This dummy/reference policy has no
                # delta accumulator to seed, so acknowledge and move on -- but it
                # MUST acknowledge: the driver blocks on the ack, and until
                # 2026-08-28 this sentinel was unknown here, fell through to
                # `num_rows = int(-7.0)` -> `continue`, and deadlocked every
                # caller that sent one (ipc_smoke2 among them).
                write_msg(fout, np.array([1.0], dtype="<f4"))
                continue
            if head == COMMIT and request.size >= 3:
                # Executed-action feedback. Nothing here caches a chunk, so there
                # is nothing to correct; acknowledge for the same reason.
                write_msg(fout, np.array([1.0], dtype="<f4"))
                continue
            if head == PLAN and request.size >= 4:
                # Accept both header lengths: pre-2026-08-28 senders end the
                # header at `dims`, newer ones append `env_id`. The payload size
                # pins which arrived (they differ by exactly one float), so this
                # is a check, not a guess. This server keeps one policy state, so
                # the id is read and ignored -- see train_eval/policy_server_exp04.py
                # for a server that routes on it.
                frames, points, dims = (int(request[1]), int(request[2]), int(request[3]))
                expected = frames * points * max(dims, 1)
                if request.size == 5 + expected:
                    flat = request[5:]
                elif request.size == 4 + expected:
                    flat = request[4:]
                else:
                    raise ValueError(
                        f"plan frame has {request.size} values; expected "
                        f"{4 + expected} (legacy) or {5 + expected} (with "
                        f"env_id) for a {frames}x{points}x{dims} plan"
                    )
                shape = (frames, points, dims) if dims else (frames, points)
                policy.reset(flat.reshape(shape).astype(np.float32))
                write_msg(fout, np.array([1.0], dtype="<f4"))  # acknowledge
                continue
            if request.size == 1 and head < 0:
                print(f"[server] shutdown after {ticks} ticks",
                      file=sys.stderr, flush=True)
                break

            num_rows = int(request[0])
            if num_rows <= 0:
                continue
            body = request[1:]
            if body.size % num_rows:
                raise ValueError(
                    f"payload {body.size} not divisible by num_rows {num_rows}"
                )
            width = body.size // num_rows

            actions = np.zeros((num_rows, action_dim), dtype=np.float32)
            for index in range(num_rows):
                row = body[index * width:(index + 1) * width]
                _, reset_flag, observation = unpack_observation(
                    row,
                    joint_dim=args.joint_dim,
                    num_points=args.num_points,
                    expect_flow=expect_flow,
                )
                # reset_flag is carried for protocol symmetry with the ACT
                # servers; the plan arrives via the PLAN frame, so honouring the
                # flag here would discard it.
                _ = reset_flag
                action = np.asarray(policy.act(observation), dtype=np.float32)
                if action.shape != (action_dim,):
                    raise ValueError(
                        f"act() returned {action.shape}, contract says ({action_dim},)"
                    )
                actions[index] = action

            write_msg(fout, actions.ravel())
            ticks += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
