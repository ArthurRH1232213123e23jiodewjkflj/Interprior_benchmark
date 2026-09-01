"""`python -m bench_cube_val ...` -> the same CLI as the console entry point.

One package per benchmark, and the package name IS the choice of benchmark:

    python -m bench_cube_val  replay --cases 8 --gpus 2,3
    python -m bench_multi_val replay --cases 8 --gpus 2,3

Both packages carry their own driver, suites, physics profiles and task
semantics, so nothing about which benchmark runs is passed as an argument —
picking the module picks the whole contract. See `api.WORKER` for why the driver
is resolved from inside the package rather than from a shared `scripts/`.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
