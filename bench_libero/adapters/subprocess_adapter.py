"""Subprocess/IPC adapter — a policy in its own process and conda env.

This is the point of the whole refactor: someone with a checkpoint that cannot
be imported alongside Isaac Sim implements one server script, and the env talks
to it over pipes using `protocol.py` (the framing already proven by the ACT and
flow servers under `_upstream/servers_ref/`).

The server is started with two FIFOs, answers a contract request, then serves
per-tick observation -> action. Reference server: js4
`000/3.eval/eval_new/policy_server_flow.py`.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .. import protocol
from ..envs.contract import PolicyContract
from .base import PolicyAdapter


class SubprocessAdapter(PolicyAdapter):
    """Drive a policy server over length-prefixed pipes."""

    def __init__(
        self,
        server_cmd: list[str],
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        pipe_dir: str | Path | None = None,
        log_path: str | Path | None = None,
        startup_timeout_s: float = 300.0,
    ) -> None:
        if not server_cmd:
            raise ValueError("server_cmd must not be empty")
        self.server_cmd = [str(x) for x in server_cmd]
        self.cwd = str(cwd) if cwd else None
        self.env = env
        self.startup_timeout_s = startup_timeout_s
        self._dir = Path(pipe_dir) if pipe_dir else Path(tempfile.mkdtemp(prefix="bench_ipc_"))
        self._in_pipe = self._dir / "env_to_policy"
        self._out_pipe = self._dir / "policy_to_env"
        self._log_path = Path(log_path) if log_path else None
        self._log = open(log_path, "ab") if log_path else subprocess.DEVNULL
        self._proc: subprocess.Popen[bytes] | None = None
        self._fin: Any = None
        self._fout: Any = None
        self._contract: PolicyContract | None = None
        self._calls = 0

    # --- lifecycle ----------------------------------------------------------
    def _spawn(self) -> None:
        """mkfifo both directions, then start the server.

        Open order matters: FIFO open() blocks until the peer connects, so the
        env opens its write end first, matching the reference driver.
        """

        # A caller-supplied pipe_dir need not exist yet -- bench_replay passes
        # <out>/_ipc, which nothing creates. Only the mkdtemp default arrives
        # ready. Without this, mkfifo raised a bare ENOENT that read like a
        # missing binary, and both smoke tests missed it because they pass
        # mkdtemp paths.
        self._dir.mkdir(parents=True, exist_ok=True)
        for pipe in (self._in_pipe, self._out_pipe):
            if not pipe.exists():
                os.mkfifo(pipe)
        cmd = self.server_cmd + [
            "--in-pipe", str(self._in_pipe),
            "--out-pipe", str(self._out_pipe),
        ]
        self._proc = subprocess.Popen(
            cmd,
            cwd=self.cwd,
            env=self.env,
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )
        self._fout = self._open_write_end()
        self._fin = self._open_read_end()

    def _server_died(self) -> str | None:
        """Exit status plus log tail, or None while the server is alive."""

        if self._proc is None:
            return "server was never started"
        code = self._proc.poll()
        if code is None:
            return None
        detail = f"policy server exited with code {code}"
        if self._log_path is not None and self._log_path.exists():
            tail = self._log_path.read_text(
                encoding="utf-8", errors="replace"
            ).strip().split("\n")[-8:]
            if tail and any(line.strip() for line in tail):
                detail += " — server log:\n  " + "\n  ".join(tail)
        return detail

    def _open_write_end(self) -> Any:
        """Open env->policy, polling so a dead server raises instead of hanging.

        O_WRONLY|O_NONBLOCK on a FIFO raises ENXIO until a reader attaches, which
        is what makes the wait interruptible.
        """

        import errno
        import time

        deadline = time.monotonic() + self.startup_timeout_s
        while True:
            died = self._server_died()
            if died is not None:
                raise ConnectionError(
                    f"policy server never connected to {self._in_pipe}: {died}"
                )
            try:
                handle = os.open(self._in_pipe, os.O_WRONLY | os.O_NONBLOCK)
            except OSError as exc:
                if exc.errno != errno.ENXIO:
                    raise
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"policy server did not open {self._in_pipe} within "
                        f"{self.startup_timeout_s:.0f}s; is it running and does it "
                        "accept --in-pipe/--out-pipe?"
                    ) from exc
                time.sleep(0.05)
                continue
            # Back to blocking: the protocol wants ordinary blocking writes.
            flags = fcntl.fcntl(handle, fcntl.F_GETFL)
            fcntl.fcntl(handle, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
            return os.fdopen(handle, "wb")

    def _open_read_end(self) -> Any:
        """Open policy->env with O_RDWR.

        O_RDONLY|O_NONBLOCK looks right and is wrong: it succeeds with no writer
        attached, and then read() returns 0 bytes — indistinguishable from a
        closed peer. The server opens its write end microseconds later, so the
        first read would spuriously report the peer as gone.

        O_RDWR on a FIFO never blocks and never yields EOF, because this fd is
        itself a writer. Reads block until real data arrives. We never write to
        it; liveness comes from `_server_died()` plus the handshake deadline.
        """

        handle = os.open(self._out_pipe, os.O_RDWR)
        return os.fdopen(handle, "rb", buffering=0)

    def _wait_readable(self, timeout_s: float) -> bool:
        """Wait for data, checking periodically whether the server is still alive.

        Needed because O_RDWR reads block forever rather than returning EOF, so a
        dead server has to be detected out of band.
        """

        import select
        import time

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ready, _, _ = select.select([self._fin], [], [], 0.5)
            if ready:
                return True
            died = self._server_died()
            if died is not None:
                raise ConnectionError(f"policy server exited while we waited: {died}")
        return False

    def handshake(self) -> PolicyContract:
        """Start the server and ask what it is."""

        if self._contract is not None:
            return self._contract
        self._spawn()
        protocol.send_contract_request(self._fout)
        # Loading a checkpoint can take a while, so the wait is generous; the
        # point is that it ends, and ends with a diagnosis.
        if not self._wait_readable(self.startup_timeout_s):
            died = self._server_died() or (
                "server is running but never answered the contract request "
                f"within {self.startup_timeout_s:.0f}s"
            )
            raise ConnectionError(f"contract handshake failed: {died}")
        try:
            payload = protocol.recv_contract(self._fin)
        except ConnectionError as exc:
            died = self._server_died() or "server alive but sent a malformed contract"
            raise ConnectionError(f"contract handshake failed: {died}") from exc
        self._contract = PolicyContract.from_dict(payload)
        return self._contract

    # --- per-tick -----------------------------------------------------------
    def step(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """One request/reply round trip.

        Row layout is the reference servers': [env_id, reset_flag, payload...].
        Single-env eval means exactly one row per tick.
        """

        assert self._contract is not None, "call handshake() first"
        row = [np.array([0.0, 0.0], dtype="<f4")]
        for key in ("robot_joint_pos", "object_surface_points", "object_flow"):
            if key in observation:
                row.append(np.asarray(observation[key], dtype="<f4").ravel())
        protocol.write_msg(self._fout, protocol.pack_rows([np.concatenate(row)]))
        reply = protocol.read_msg(self._fin)
        if reply is None:
            died = self._server_died() or "pipe closed while the server was alive"
            raise ConnectionError(f"policy server stopped mid-episode: {died}")
        self._calls += 1
        return protocol.unpack_reply(reply, 1, self._contract.action_dim)[0]

    def step_batch(self, observations: list[dict[str, np.ndarray]]) -> np.ndarray:
        """One round trip for N envs. Returns (N, action_dim).

        Row `i` carries `env_id=i` and the reply must come back in the same
        order. The id is redundant with position here, but it is what a server
        keys its per-env state on, so it is sent explicitly rather than left for
        the server to infer.
        """

        assert self._contract is not None, "call handshake() first"
        rows = []
        for index, observation in enumerate(observations):
            row = [np.array([float(index), 0.0], dtype="<f4")]
            for key in ("robot_joint_pos", "object_surface_points", "object_flow"):
                if key in observation:
                    row.append(np.asarray(observation[key], dtype="<f4").ravel())
            rows.append(np.concatenate(row))
        widths = {row.size for row in rows}
        if len(widths) != 1:
            raise ValueError(
                f"observation rows have mismatched widths {widths}; the server "
                "cannot split a ragged batch"
            )

        protocol.write_msg(self._fout, protocol.pack_rows(rows))
        reply = protocol.read_msg(self._fin)
        if reply is None:
            died = self._server_died() or "pipe closed while the server was alive"
            raise ConnectionError(f"policy server stopped mid-episode: {died}")
        self._calls += 1
        return protocol.unpack_reply(reply, len(rows), self._contract.action_dim)

    def commit_action(
        self,
        model_action: np.ndarray,
        *,
        chunk_invalidation_tolerance_rad: float = 0.0,
    ) -> None:
        """Tell the server what was ACTUALLY commanded after env safety edits.

        This used to inherit the no-op base method, so nothing crossed the wire
        and a chunking policy's notion of the executed target silently diverged
        from the simulator's. For exp04 the effect was total: inference ran on
        647 of 647 ticks instead of ~21, no cached chunk ever executed to
        completion, and the grasp never finished.

        Frame layout mirrors a tick: [COMMIT, num_rows, tolerance] ++ rows of
        action_dim. The server acks, so a desync surfaces here instead of as a
        mysteriously bad score.
        """

        assert self._contract is not None, "call handshake() first"
        actions = np.atleast_2d(np.asarray(model_action, dtype="<f4"))
        if actions.shape[-1] != self._contract.action_dim:
            raise ValueError(
                f"commit_action got width {actions.shape[-1]}, contract says "
                f"{self._contract.action_dim}"
            )
        payload = np.concatenate((
            np.array([protocol.COMMIT, float(actions.shape[0]),
                      float(chunk_invalidation_tolerance_rad)], dtype="<f4"),
            actions.astype("<f4").ravel(),
        ))
        protocol.write_msg(self._fout, payload)
        ack = protocol.read_msg(self._fin)
        if ack is None:
            died = self._server_died() or "pipe closed during commit_action"
            raise ConnectionError(f"policy server rejected the commit: {died}")

    def reset(
        self,
        robot_joint_pos: np.ndarray,
        *,
        object_flow_plan: np.ndarray | None = None,
        env_id: int = protocol.BROADCAST_ENV_ID,
    ) -> None:
        """Start an episode for ONE env, optionally handing over its flow plan.

        For full_plan conditioning this is the only time the policy sees the
        plan — that is the task: grasp the cube, then follow this.

        `env_id` addresses which env this episode belongs to. It matters whenever
        a shard runs N cases as N parallel envs: each env has its OWN plan and
        its OWN initial joints, and a server holding one shared runner would
        otherwise have every env chase case 0's target (defect A, log/0828.md).
        The default -1 broadcasts, which is the correct single-env behaviour.
        """

        assert self._contract is not None, "call handshake() first"
        # Send the RECORDED initial joint position, not nothing. Until 2026-08-27
        # this argument was accepted and dropped, so a server could only use the
        # joints it read on its first tick -- i.e. post-settle simulator values.
        # For a delta-action policy (exp04 is arm_delta_hand_absolute) that is the
        # base every subsequent delta accumulates from, and upstream's own driver
        # seeds it from `episode.arrays["robot_joint_pos"][start]`.
        protocol.send_init_state(
            self._fout, robot_joint_pos, env_id=env_id
        )
        ack = protocol.read_msg(self._fin)
        if ack is None:
            died = self._server_died() or "pipe closed during init-state delivery"
            raise ConnectionError(f"policy server rejected the initial state: {died}")

        if object_flow_plan is None:
            protocol.send_reset(self._fout)
            return
        protocol.send_plan(self._fout, object_flow_plan, env_id=env_id)
        ack = protocol.read_msg(self._fin)
        if ack is None:
            died = self._server_died() or "pipe closed during plan delivery"
            raise ConnectionError(f"policy server rejected the plan: {died}")

    def close(self) -> None:
        """Send shutdown, then reap. Never leave a server holding a GPU."""

        try:
            if self._fout is not None:
                protocol.send_shutdown(self._fout)
        except (BrokenPipeError, OSError):
            pass
        for handle in (self._fin, self._fout):
            try:
                if handle is not None:
                    handle.close()
            except OSError:
                pass
        self._fin = self._fout = None
        if self._proc is not None:
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if self._log not in (None, subprocess.DEVNULL):
            self._log.close()
        for pipe in (self._in_pipe, self._out_pipe):
            pipe.unlink(missing_ok=True)

    # --- diagnostics --------------------------------------------------------
    @property
    def inference_calls(self) -> int:
        return self._calls

    def stats(self) -> dict[str, Any]:
        return {
            "action_source": "subprocess_ipc",
            "server_cmd": " ".join(self.server_cmd),
            "pipe_dir": str(self._dir),
            # Round trips, not rows: one call carries every env's row. The
            # server reports rows and per-env rates in its own stats line.
            "round_trips": self._calls,
        }
