"""Rotation helpers, vendored so the env needs no `flow_policy` import.

`rot6d_to_rotation_matrix` is a verbatim port of
`flow_policy/actions.py:134`; `rotation_matrix_to_quat_wxyz` of
`sim_rollout_render.py:368` (`_rotation_matrix_to_quat_wxyz`).  Keep them in
lockstep with upstream — an EE-space policy's IK target depends on both.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def rot6d_to_rotation_matrix(rot6d: Any) -> Any:
    """Decode ``[..., 6]`` into a valid rotation matrix ``[..., 3, 3]``.

    Partial Gram-Schmidt over the six values read as two R^3 vectors, then a
    cross product to complete the frame.  Always orthonormal with determinant
    +1 for non-degenerate input; degenerate input raises rather than silently
    collapsing to a reflection.
    """

    import torch

    if rot6d.shape[-1] != 6:
        raise ValueError(f"rot6d must have shape [..., 6]; got {tuple(rot6d.shape)}")
    raw_first = rot6d[..., :3]
    raw_second = rot6d[..., 3:]
    first_norm = torch.linalg.vector_norm(raw_first, dim=-1, keepdim=True)
    if bool(torch.any(first_norm <= 0)):
        raise ValueError("rot6d first vector has zero norm; cannot form a rotation")
    first_column = raw_first / first_norm
    projection = (first_column * raw_second).sum(dim=-1, keepdim=True)
    residual = raw_second - projection * first_column
    residual_norm = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
    if bool(torch.any(residual_norm <= 0)):
        raise ValueError(
            "rot6d second vector is colinear with the first; cannot form a rotation"
        )
    second_column = residual / residual_norm
    third_column = torch.linalg.cross(first_column, second_column, dim=-1)
    return torch.stack([first_column, second_column, third_column], dim=-1)


def rotation_matrix_to_quat_wxyz(rotation: Any) -> np.ndarray:
    """Convert one proper 3x3 rotation matrix to a normalized wxyz quaternion."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation must have shape [3,3], got {matrix.shape}")
    quaternion = np.empty(4, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion[0] = 0.25 * scale
        quaternion[1] = (matrix[2, 1] - matrix[1, 2]) / scale
        quaternion[2] = (matrix[0, 2] - matrix[2, 0]) / scale
        quaternion[3] = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quaternion[:] = (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quaternion[:] = (
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quaternion[:] = (
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
    quaternion /= max(float(np.linalg.norm(quaternion)), 1.0e-12)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion.astype(np.float32)
