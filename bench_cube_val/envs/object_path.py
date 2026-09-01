"""Alternate object-path sources (optional).

The default plan comes straight from the derived shard, so this module is only
touched when a contract declares `object_path_source` (e.g.
`change_only_geometric_path`).

Upstream's loader lives in `flow_policy/datasets/change_only_paths.py` — 459
lines coupled to `TroMpZarrDataset`, `..conditioning`, and
`..flow_representation`.  Vendoring that would drag the training package into
the benchmark for an optional path, so it is imported lazily instead: the
dependency exists only for policies that ask for it, and `bench` stays
importable without `flow_policy` on the path.

To evaluate such a policy without the training package, either pre-bake the
path into the shard or reimplement `load_change_only_path_poses` here against
the archive format alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CHANGE_ONLY_GEOMETRIC_PATH = "change_only_geometric_path"
"""Upstream `flow_policy.datasets` constant, vendored."""


def load_change_only_path_poses(
    archive: Path,
    *,
    expected_manifest_sha256: str,
    shard_dir: Path,
    episode: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (path_points, path_poses) for a change-only geometric path.

    Delegates to `flow_policy`; raises a directed error when it is absent.
    """

    try:
        from flow_policy.datasets import (
            load_change_only_path_poses as _upstream_loader,
        )
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "object_path_source=change_only_geometric_path needs flow_policy on "
            "the path (it owns the archive format). Either run this adapter in "
            "the training env, pre-bake the path into the shard, or drop "
            "object_path_source from the contract."
        ) from exc

    path = archive if archive.is_absolute() else Path.cwd() / archive
    return _upstream_loader(
        path,
        expected_manifest_sha256=expected_manifest_sha256,
        shard_dir=shard_dir,
        episode=episode,
    )
