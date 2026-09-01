"""Wire protocol for the env/policy boundary.

Ported, not designed.  The framing and message shapes come from
`_upstream/servers_ref/` and js4 `000/3.eval/eval_new/policy_server_flow.py`,
whose docstring is the authoritative spec:

    >I byte-length header, <f4 payload

    handshake (once, before any tick):
      [-2.0, num_envs, N] ++ per env: [env_id, points_obj(N*3)]
      -> reply [1.0] as ack

    per tick:
      [num_rows] ++ per row: [env_id, reset_flag, ...payload...]
      -> reply [num_rows * action_dim]

    control:
      size 0          -> ignored (stateless policies)
      size 1, value<0 -> shutdown

`read_msg` / `write_msg` below are byte-for-byte the upstream implementations
(`policy_server_flow.py:97-116`, identical in `servers/policy_server.py:40-58`
and the grasp_follow driver at :20-26).  Do not "improve" them: both sides of
the existing ACT and flow servers already speak exactly this, so any change
breaks compatibility with servers we did not write.

The one addition is `send_contract` / `recv_contract`: a JSON handshake frame
carrying `PolicyContract`, replacing the attribute reads the in-process runner
used to answer.  It rides the same length-prefixed framing but with a UTF-8
JSON body, distinguished by the CONTRACT sentinel.

Per-env addressing
------------------
The per-tick rows have always carried `env_id` (upstream's layout).  The frames
we added -- `PLAN` and `INIT_STATE` -- did not, because their first use was a
single-env eval.  That gap is what forced the driver to install only case 0's
plan when a shard ran N cases in N parallel envs, so N-1 envs chased case 0's
target (`log/0828.md`).

`env_id` is therefore appended to the END of those two headers, with `-1`
meaning "broadcast to every env" -- exactly the old single-env semantics.
Appending keeps the existing header indices valid, and the parsers below accept
BOTH the old and the new header: the payload length pins which one arrived
(old = 4 + frames*points*dims, new = 5 + that), so there is no ambiguity to
guess at.  Both parsers live here so the sender and receiver cannot drift --
a mis-parsed plan is not an error, it is a plausible wrong score.
"""

from __future__ import annotations

import json
import struct
from typing import Any, BinaryIO

import numpy as np

# --- sentinels (first payload value) ----------------------------------------
HANDSHAKE = -2.0
"""Upstream's canonical points_obj handshake."""

SHUTDOWN = -1.0
"""size-1 frame with value<0 -> server exits."""

CONTRACT = -3.0
"""Ours: next frame is a UTF-8 JSON PolicyContract."""

RESET = -4.0
"""Ours: begin a fresh episode, no plan attached."""

PLAN = -5.0
"""Ours: [-5.0, frames, points, dims, env_id] ++ flat plan; reply [1.0] to ack.

`env_id` is the trailing header value; -1 broadcasts to every env. Before
2026-08-28 the header stopped at `dims` and the plan could only ever address one
env, which is the root of defect A in `log/0828.md`.
"""

#: Recorded initial joint position: [-7.0, dim, env_id] ++ joints. Sent at reset,
#: before the plan. A delta-action policy accumulates from this base, and the
#: simulator's post-settle readback is not the same value the recording started
#: from. `env_id` -1 broadcasts.
INIT_STATE = -7.0

#: Executed-action feedback: [-6.0, num_rows, tolerance] ++ rows of action_dim.
#: The env clamps a policy's output to joint limits and a max per-tick step, so
#: what PhysX executes can differ from what `step()` returned. A chunking policy
#: that is not told the real command drops its cached chunk every tick.
#: Row i addresses env i positionally -- the same convention as a per-tick batch
#: -- so this frame needs no explicit env_id.
COMMIT = -6.0

#: Broadcast target for the per-env frames: apply to every env the server knows.
#: This is the pre-2026-08-28 single-env behaviour, kept as the default so a
#: single-env caller need not care about env ids at all.
BROADCAST_ENV_ID = -1

ACK = np.array([1.0], dtype="<f4")


# --- framing (verbatim upstream) --------------------------------------------
def read_msg(f: BinaryIO) -> np.ndarray | None:
    """Read one length-prefixed float32 frame. None on clean EOF."""

    hdr = f.read(4)
    if len(hdr) < 4:
        return None
    n = struct.unpack(">I", hdr)[0]
    if n == 0:
        return np.empty(0, dtype=np.float32)
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return np.frombuffer(buf, dtype="<f4")


def write_msg(f: BinaryIO, arr: Any) -> None:
    """Write one length-prefixed float32 frame and flush."""

    arr = np.asarray(arr, dtype="<f4")
    f.write(struct.pack(">I", arr.nbytes))
    f.write(arr.tobytes())
    f.flush()


# --- JSON frames (contract handshake) ---------------------------------------
def write_json(f: BinaryIO, payload: dict[str, Any]) -> None:
    """Same header, UTF-8 JSON body."""

    body = json.dumps(payload).encode("utf-8")
    f.write(struct.pack(">I", len(body)))
    f.write(body)
    f.flush()


def read_json(f: BinaryIO) -> dict[str, Any] | None:
    hdr = f.read(4)
    if len(hdr) < 4:
        return None
    n = struct.unpack(">I", hdr)[0]
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return json.loads(buf.decode("utf-8"))


def send_contract_request(fout: BinaryIO) -> None:
    """Env -> server: declare your contract."""

    write_msg(fout, np.array([CONTRACT], dtype="<f4"))


def recv_contract(fin: BinaryIO) -> dict[str, Any]:
    """Env <- server: the PolicyContract as a dict."""

    payload = read_json(fin)
    if payload is None:
        raise ConnectionError("policy server closed during contract handshake")
    return payload


def send_shutdown(fout: BinaryIO) -> None:
    write_msg(fout, np.array([SHUTDOWN], dtype="<f4"))


def send_reset(fout: BinaryIO) -> None:
    """Episode boundary for a policy that consumes no plan."""

    write_msg(fout, np.array([RESET], dtype="<f4"))


def send_plan(
    fout: BinaryIO,
    plan: np.ndarray,
    *,
    env_id: int = BROADCAST_ENV_ID,
) -> None:
    """Hand over the immutable object-flow plan for one env (or all).

    Shape travels in the header values because the payload is flat: a
    [frames, points, 3] plan is 768*1024*3 floats and the server has to know how
    to fold it. `env_id` rides at the end of the header; -1 broadcasts.
    """

    array = np.asarray(plan, dtype="<f4")
    shape = list(array.shape) + [0.0] * (3 - array.ndim)
    header = np.array([PLAN, *shape[:3], float(env_id)], dtype="<f4")
    write_msg(fout, np.concatenate((header, array.ravel())))


def recv_plan(request: np.ndarray) -> tuple[np.ndarray, int]:
    """Parse a PLAN frame -> (plan, env_id). The inverse of `send_plan`.

    Accepts both header lengths. The payload size decides which arrived:
    `frames*points*dims` floats follow a 4-value header (pre-2026-08-28, no
    env_id) or a 5-value one (with env_id). The two differ by exactly one float,
    so this is a check rather than a guess -- and a genuinely malformed frame
    still raises instead of being folded to a wrong shape.
    """

    if request.size < 4 or float(request[0]) != PLAN:
        raise ValueError("not a PLAN frame")
    frames, points, dims = int(request[1]), int(request[2]), int(request[3])
    expected = frames * points * max(dims, 1)
    shape = (frames, points, dims) if dims else (frames, points)

    if request.size == 5 + expected:          # new: env_id present
        env_id = int(request[4])
        flat = request[5:]
    elif request.size == 4 + expected:        # legacy sender: broadcast
        env_id = BROADCAST_ENV_ID
        flat = request[4:]
    else:
        raise ValueError(
            f"plan frame has {request.size} values; expected "
            f"{4 + expected} (legacy) or {5 + expected} (with env_id) for a "
            f"{frames}x{points}x{dims} plan"
        )
    return flat.reshape(shape).astype(np.float32), env_id


def send_init_state(
    fout: BinaryIO,
    joints: np.ndarray,
    *,
    env_id: int = BROADCAST_ENV_ID,
) -> None:
    """Hand over the RECORDED initial joint position for one env (or all).

    This is the base a delta-action policy accumulates every delta from, so it
    must be the recording's value at `start_frame`, not the simulator's
    post-settle readback.
    """

    array = np.asarray(joints, dtype="<f4").ravel()
    header = np.array([INIT_STATE, float(array.size), float(env_id)], dtype="<f4")
    write_msg(fout, np.concatenate((header, array)))


def recv_init_state(request: np.ndarray) -> tuple[np.ndarray, int]:
    """Parse an INIT_STATE frame -> (joints, env_id).

    Accepts both header lengths, the same way `recv_plan` does.
    """

    if request.size < 2 or float(request[0]) != INIT_STATE:
        raise ValueError("not an INIT_STATE frame")
    dim = int(request[1])
    if request.size == 3 + dim:            # new: env_id present
        env_id = int(request[2])
        joints = request[3:]
    elif request.size == 2 + dim:           # legacy sender: broadcast
        env_id = BROADCAST_ENV_ID
        joints = request[2:]
    else:
        raise ValueError(
            f"init-state frame has {request.size} values; expected "
            f"{2 + dim} (legacy) or {3 + dim} (with env_id) for dim={dim}"
        )
    return joints.astype(np.float32), env_id


# --- row packing ------------------------------------------------------------
def pack_rows(rows: list[np.ndarray]) -> np.ndarray:
    """[num_rows] ++ concatenated rows, as the per-tick request."""

    flat = [np.asarray(row, dtype="<f4").ravel() for row in rows]
    return np.concatenate(
        [np.array([len(flat)], dtype="<f4"), *flat] if flat
        else [np.array([0], dtype="<f4")]
    )


def unpack_reply(reply: np.ndarray, num_rows: int, action_dim: int) -> np.ndarray:
    """Reply -> (num_rows, action_dim); raises on a width mismatch."""

    expected = num_rows * action_dim
    if reply.size != expected:
        raise ValueError(
            f"policy reply has {reply.size} values, expected {expected} "
            f"({num_rows} rows x {action_dim})"
        )
    return reply.reshape(num_rows, action_dim).astype(np.float32)
