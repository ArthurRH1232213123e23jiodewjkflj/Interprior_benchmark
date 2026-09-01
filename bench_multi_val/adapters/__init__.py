"""Adapter registry — resolves a name to a PolicyAdapter implementation.

`build_adapter` is the single seam the env calls instead of constructing a
`FlowPolicyRunner` inline.  The default is the flow_policy reference adapter,
so existing invocations behave exactly as before; a foreign checkpoint selects
another adapter (typically `subprocess`, which speaks `protocol.py` to a server
running in its own conda env).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import PolicyAdapter

DEFAULT_ADAPTER = "flow_policy_v10"


def build_adapter(
    ckpt: str | Path,
    model_config: str | Path,
    *,
    kind: str = DEFAULT_ADAPTER,
    **kwargs: Any,
) -> PolicyAdapter:
    """Instantiate one adapter by name.

    Imports are lazy so that selecting `subprocess` never pulls `flow_policy`
    into the process, and vice versa.
    """

    if kind in ("flow_policy_v10", "flow_policy", "default"):
        from .flow_policy_v10 import FlowPolicyV10Adapter

        return FlowPolicyV10Adapter(ckpt, model_config, **kwargs)

    if kind in ("subprocess", "ipc", "server"):
        from .subprocess_adapter import SubprocessAdapter

        return SubprocessAdapter(**kwargs)

    if kind in ("recorded_teacher", "teacher"):
        from .recorded_teacher import RecordedTeacherAdapter

        return RecordedTeacherAdapter(**kwargs)

    raise ValueError(
        f"unknown adapter kind {kind!r}; expected one of: flow_policy_v10, "
        "subprocess, recorded_teacher"
    )


__all__ = ["PolicyAdapter", "build_adapter", "DEFAULT_ADAPTER"]
