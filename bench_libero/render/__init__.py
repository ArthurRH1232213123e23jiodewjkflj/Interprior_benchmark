"""bench_libero rendering.

robot_meshes      shared xArm7+Wuji geometry (FK from 27 joints, or recorded poses).
                  The one place robot geometry comes from -- see its docstring for why.
libero_task_viewer  static HTML for one LIBERO case: real mesh + real arm + goal path.
"""
from . import robot_meshes  # noqa: F401
