"""bench_libero — evaluate a manipulation policy against TRO-MP.

    from bench_libero import run

    report = run(
        policy="/your/env/bin/python your_server.py --ckpt model.pt",
        suite="cube_lift_follow_v1",
        cases=8, html_cases=2, gpus=[6, 7],
    )

Your checkpoint stays in your own process, your own conda env, your own
framework. See scripts/policy_server_reference.py — you edit two functions in a
copy of that file and nothing else.
"""

from .api import DEFAULT_INTERPRIOR_ROOT, DEFAULT_ISAAC_PYTHON, Report, run

__all__ = ["run", "Report", "DEFAULT_ISAAC_PYTHON", "DEFAULT_INTERPRIOR_ROOT"]
