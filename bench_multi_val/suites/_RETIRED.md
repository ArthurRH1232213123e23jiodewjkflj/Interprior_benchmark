# _retired_cube_lift_follow_v1.yaml.bak_wrong_pkg

`bench_multi_val` was copied from `bench_cube_val`, so it arrived carrying the
cube suite verbatim (sha d8c4367743467509 -- byte-identical to
`bench_cube_val/suites/cube_lift_follow_v1.yaml`, which is where it still lives
and still runs).

Retired here rather than kept, because inside this package it can only mislead:
it pins the cube profile and the venv_share checkout, so `bench_multi_val tasks`
listed a suite that this package's gate refuses to run -- profile sha and
interprior_root both fail against the Objaverse baseline. That refusal is the
correct behaviour, and is what `BASELINE_ADVISORY_ONLY = False` is for, but a
suite that is listed yet unrunnable invites someone to "fix" it by relaxing the
gate.

Run the cube benchmark with `python -m bench_cube_val`.
