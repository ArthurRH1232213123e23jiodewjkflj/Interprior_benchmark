"""Public API — one call runs a checkpoint through the benchmark.

    from bench_multi_val import run

    report = run(
        policy="/your/conda/env/bin/python your_server.py --ckpt model.pt",
        suite="cube_lift_follow_v1",
        cases=8,
        html_cases=2,
        gpus=[6, 7],
    )
    print(report.reference_success, report.guide_tracking_mean_m)

Everything below is orchestration. The evaluation itself lives in
`driver.py`, which owns the physics gate, state injection, ZOH,
the v10 metric set and the Three.js pages — one worker for both teacher replay
and policy eval, because the only difference is where actions come from.

Why subprocesses per GPU rather than one process with many envs: Isaac Lab binds
a process to one CUDA device at import, so multi-GPU means multi-process. That is
also what upstream's `eval_rollout.py` does, and it gives crash isolation for
free — one shard dying leaves the others' results on disk.

Resume is by artifact, not by return code. Isaac's Kit shutdown path can mask a
Python exception as exit 0, so a shard counts as done only when its summary.json
exists. That bit us during development: a run reported success with an empty
output directory.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

PACKAGE_DIR = Path(__file__).resolve().parent
BENCH_ROOT = PACKAGE_DIR.parent

# The driver lives INSIDE the package, and `WORKER` is derived from this file's
# own location rather than from a shared `scripts/` directory. That is the whole
# point of one package per benchmark: `bench_multi_val` and `bench_multi_val` are
# copies whose task semantics (four-class thresholds, the v10 metric set, state
# injection) diverge, so a shared worker path would make one package silently
# score with the other's driver — the gate would still pass, against the wrong
# baseline. Same reasoning as the profile path in the driver itself.
WORKER = PACKAGE_DIR / "driver.py"
SUITES_DIR = PACKAGE_DIR / "suites"

# Candidates in priority order — all of them ours. There is deliberately no
# fallback into another user's home: such a path is a symlink its owner can
# repoint, so a run could silently use a different Isaac build than the one we
# validated, and the warning would scroll past unread.
_ISAAC_PYTHON_CANDIDATES = (
    "~/pro5000_env/.venv_isaacsim_pro5000/bin/python",
    "~/Interprior_bench_env/.venv_isaacsim_pro5000/bin/python",
    "~/code/Interprior/.venv_isaacsim_pro5000/bin/python",
)

BUILD_HINT = (
    "Build one (README):\n"
    "  cp -a /mnt/venv_share/pro5000/Interprior ~/pro5000_env\n"
    "  CODE_DIR=~/pro5000_env bash "
    "/mnt/venv_share/pro5000/build_pro5000_venv_isaacsim.sh\n"
    "Or point at an existing one:\n"
    "  export INTERPRIOR_BENCH_ISAAC_PYTHON=/path/to/.venv/bin/python\n"
    "Note: uv is only installed on the Pro5000 nodes, so build there."
)

# The env code is the other half of physics truth and must come from the same
# checkout as the frozen task yaml. The shared mounts are root-owned and
# node-shared, so falling back to them is correct, not a compromise.
# See log/physics_difference.txt.
# Unlike the venv, these fallbacks are legitimate: /mnt/venv_share is root-owned,
# node-shared, and IS the physics truth (sha256 e9de8133). A local copy still
# comes first so the task yaml and the env code sit together under our control.
_INTERPRIOR_ROOT_CANDIDATES = (
    "~/pro5000_env",
    "~/Interprior_bench_env",
    "/mnt/venv_share/pro5000/Interprior",
    "/mnt/venv_share/H800/Interprior",
)

def _home() -> Path:
    """This user's home — NOT `~`.

    /etc/environment on these nodes hardcodes HOME=/root, so `~` expands to a
    directory we neither own nor can read. The passwd entry for our own uid is
    the only reliable source; $HOME is used only when it really is ours.
    """

    import pwd

    try:
        entry = pwd.getpwuid(os.getuid())
        if entry.pw_dir and entry.pw_dir not in ("/root", "/"):
            return Path(entry.pw_dir)
    except (KeyError, OSError):
        pass
    env_home = os.environ.get("HOME", "")
    if env_home and env_home not in ("/root", "/"):
        return Path(env_home)
    return Path.cwd()


def _expand(candidate: str) -> Path:
    """Expand a leading ~ against the real home."""

    if candidate.startswith("~/"):
        return _home() / candidate[2:]
    if candidate == "~":
        return _home()
    return Path(candidate)


def _exists(path: Path, *, want_file: bool) -> bool:
    """Probe without raising.

    stat() throws PermissionError on a path we cannot traverse, and one
    unreadable candidate must not abort resolution — unreadable means "not this
    one", not "give up".
    """

    try:
        return path.is_file() if want_file else path.is_dir()
    except (OSError, PermissionError):
        return False


def _resolve_first(candidates: tuple[str, ...], env_var: str, *, want_file: bool) -> str:
    """First candidate that exists.

    Falls back to the last candidate rather than raising, so a missing
    environment produces an error naming a real path instead of an empty string.
    """

    override = os.environ.get(env_var)
    if override:
        return override
    for candidate in candidates:
        path = _expand(candidate)
        if _exists(path, want_file=want_file):
            return str(path)
    return str(_expand(candidates[-1]))


def resolve_isaac_python(*, required: bool = False) -> str:
    """Locate an Isaac python among our own candidates.

    `required=True` raises when none exists — better than handing subprocess a
    path that is not there and reading the failure out of a worker log later.
    """

    chosen = _resolve_first(
        _ISAAC_PYTHON_CANDIDATES, "INTERPRIOR_BENCH_ISAAC_PYTHON", want_file=True
    )
    if required and not _exists(Path(chosen), want_file=True):
        tried = "\n  ".join(str(_expand(c)) for c in _ISAAC_PYTHON_CANDIDATES)
        raise FileNotFoundError(
            f"no Isaac python found. Tried:\n  {tried}\n\n{BUILD_HINT}"
        )
    return chosen


def resolve_interprior_root() -> str:
    """Locate the env code that pairs with the frozen physics profile."""

    return _resolve_first(
        _INTERPRIOR_ROOT_CANDIDATES, "INTERPRIOR_BENCH_ROOT", want_file=False
    )


DEFAULT_ISAAC_PYTHON = resolve_isaac_python()
DEFAULT_INTERPRIOR_ROOT = resolve_interprior_root()


@dataclass
class Report:
    """Merged result across every shard."""

    suite: str
    cases: int
    frames: int
    mode: str
    out_dir: Path

    # --- v10 / zjw headline counts (comparable to any existing summary.json) --
    ever_lifted: int = 0
    dropped_after_lift: int = 0
    lift_and_hold: int = 0
    lift_and_follow: int = 0
    reference_success: int = 0
    demo_success: int = 0

    guide_tracking_mean_m: float = 0.0
    guide_tracking_fraction_within_3cm: float = 0.0
    guide_tracking_fraction_within_5cm: float = 0.0

    # --- physics faithfulness (teacher replay only) --------------------------
    pos_err_median_mm: float | None = None
    physics_gate_ok: bool | None = None
    physics_profile_sha256: str = ""

    class_counts: dict[str, int] = field(default_factory=dict)
    cases_detail: list[dict[str, Any]] = field(default_factory=list)
    html_pages: list[dict[str, Any]] = field(default_factory=list)
    shards: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        """Every shard produced its artifact."""

        return not self.failures

    def __str__(self) -> str:
        lines = [
            f"{self.suite}: {self.cases} cases x {self.frames} frames ({self.mode})",
            f"  ever_lifted        {self.ever_lifted}/{self.cases}",
            f"  dropped_after_lift {self.dropped_after_lift}",
            f"  lift_and_follow    {self.lift_and_follow}/{self.cases}",
        ]
        key = "demo_success" if self.mode == "policy_eval" else "reference_success"
        lines.append(f"  {key:18s} {getattr(self, key)}/{self.cases}")
        lines.append(
            f"  guide tracking     mean {self.guide_tracking_mean_m * 1000:.1f} mm, "
            f"within5cm {self.guide_tracking_fraction_within_5cm * 100:.1f}%"
        )
        if self.pos_err_median_mm is not None:
            lines.append(f"  replay pos error   median {self.pos_err_median_mm:.2f} mm")
        if self.class_counts:
            lines.append(f"  four-class         {self.class_counts}")
        if self.failures:
            lines.append(f"  FAILED shards      {len(self.failures)}")
        if self.html_pages:
            lines.append(f"  viewer pages       {len(self.html_pages)}")
        return "\n".join(lines)


def run(
    *,
    policy: str | None = None,
    suite: str | Path = "cube_lift_follow_v1",
    cases: int | None = None,
    frames: int | None = 0,
    html_cases: int = 0,
    html_frames: int = 300,
    capture_stride: int = 4,
    gpus: Sequence[int] = (0,),
    out: str | Path | None = None,
    isaac_python: str = DEFAULT_ISAAC_PYTHON,
    interprior_root: str | Path | None = None,
    pos_tol_mm: float = 3.0,
    reach_tol_cm: float = 3.0,
    max_target_step_rad: float | None = None,
    chunk_invalidation_tolerance_rad: float | None = None,
    resume: bool = True,
    env_extra: dict[str, str] | None = None,
) -> Report:
    """Evaluate a policy — or, with `policy=None`, replay teacher actions.

    policy          command that starts your policy server; it will be given
                    --in-pipe/--out-pipe. None runs the physics gate instead.
    suite           name under `bench_multi_val/suites/` or a path to a yaml
    cases           how many cases to evaluate (None keeps the suite's own limit)
    frames          0 = the full recording (2400); None = the 768 training
                    window; N = the first N frames
    html_cases      how many cases get a Three.js rollout page (0 none, -1 all).
                    Each page is tens of MB, so this is opt-in.
    html_frames     max playback frames per page
    gpus            one worker process per GPU; cases are split across them
    resume          skip cases whose summary.json already exists

    The task, for reference: each case starts from a recorded initial state with
    the object on the table and hands the policy that episode's whole object-flow
    plan at reset. The policy must grasp the cube and then follow the plan.
    """

    # Fail here, not inside a worker: a missing venv should name the build
    # command, not surface as an exec error in gpu0.log.
    if isaac_python == DEFAULT_ISAAC_PYTHON:
        isaac_python = resolve_isaac_python(required=True)

    suite_path = _resolve_suite(suite)

    # The SUITE owns interprior_root, because env code is half of physics truth
    # and has to pair with that suite's frozen profile. The module-level
    # candidate scan cannot know which benchmark is running, and this package's
    # suites need a checkout that declares the heterogeneous object-pool fields
    # (scene_utils.py:2000) -- the venv_share candidates predate them, so taking
    # the scanned default would spawn one shared object for every case, or now,
    # with a fatal gate, refuse to run at all.
    #
    # Precedence: explicit argument > suite declaration > scanned candidates.
    if interprior_root is None:
        interprior_root = _suite_interprior_root(suite_path) or DEFAULT_INTERPRIOR_ROOT
    out_dir = Path(out).expanduser().resolve() if out else _default_out(suite_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    total = _count_cases(suite_path, cases)
    if total == 0:
        raise ValueError(f"suite {suite_path} resolved to zero cases")
    gpu_list = list(gpus) or [0]
    shards = _split(total, len(gpu_list))

    # Viewer pages are assigned to the first shard only: a page is tens of MB and
    # `html_cases` means "give me a few to look at", not "a few per GPU".
    html_budget = total if html_cases < 0 else min(html_cases, total)

    print(f"[bench] suite={suite_path.stem} cases={total} gpus={gpu_list} "
          f"mode={'policy_eval' if policy else 'teacher_replay'}", flush=True)
    if html_budget:
        print(f"[bench] viewer pages: {html_budget} (first shard)", flush=True)

    started = time.time()
    failures: list[dict[str, Any]] = []
    lock = threading.Lock()
    threads = []
    remaining_html = html_budget

    for gpu, indices in zip(gpu_list, shards):
        if not indices:
            continue
        shard_html = min(remaining_html, len(indices))
        remaining_html -= shard_html
        thread = threading.Thread(
            target=_run_shard,
            args=(gpu, indices, shard_html),
            kwargs=dict(
                policy=policy, suite_path=suite_path, out_dir=out_dir,
                cases=cases, frames=frames, html_frames=html_frames,
                capture_stride=capture_stride, isaac_python=isaac_python,
                interprior_root=interprior_root, pos_tol_mm=pos_tol_mm,
                reach_tol_cm=reach_tol_cm,
                max_target_step_rad=max_target_step_rad,
                chunk_invalidation_tolerance_rad=chunk_invalidation_tolerance_rad,
                resume=resume,
                env_extra=env_extra, failures=failures, lock=lock,
            ),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    report = _merge(out_dir, suite_path, total, failures, time.time() - started)
    (out_dir / "report.json").write_text(
        json.dumps(_report_dict(report), indent=2), encoding="utf-8"
    )
    print(f"\n{report}", flush=True)
    print(f"[bench] {out_dir}/report.json", flush=True)
    return report


def _run_shard(
    gpu: int,
    indices: list[int],
    shard_html: int,
    *,
    policy: str | None,
    suite_path: Path,
    out_dir: Path,
    cases: int | None,
    frames: int | None,
    html_frames: int,
    capture_stride: int,
    isaac_python: str,
    interprior_root: str | Path,
    pos_tol_mm: float,
    reach_tol_cm: float,
    max_target_step_rad: float | None,
    chunk_invalidation_tolerance_rad: float | None,
    resume: bool,
    env_extra: dict[str, str] | None,
    failures: list[dict[str, Any]],
    lock: threading.Lock,
) -> None:
    shard_dir = out_dir / f"gpu{gpu}"
    summary = shard_dir / "summary.json"
    if resume and summary.is_file():
        print(f"[gpu{gpu}] already done, skipping {len(indices)} cases", flush=True)
        return
    shard_dir.mkdir(parents=True, exist_ok=True)

    command = [
        isaac_python, "-u", str(WORKER),
        "--suite", str(suite_path),
        "--interprior-root", str(interprior_root),
        "--out", str(shard_dir),
        "--case-indices", ",".join(str(i) for i in indices),
        "--html-cases", str(shard_html),
        "--html-frames", str(html_frames),
        "--capture-stride", str(capture_stride),
        "--pos-tol-mm", str(pos_tol_mm),
        "--reach-tol-cm", str(reach_tol_cm),
    ]
    if frames is not None:
        command += ["--frames", str(frames)]
    if cases is not None:
        command += ["--limit", str(cases)]
    # Policy-side inference knobs. Omitted -> the driver's own defaults, which are
    # conservative on purpose. A chunking policy generally needs both raised:
    # max_target_step_rad 0.05 clips every step (zjw's eval uses 100.0 and still
    # logged 333 clips), and the chunk tolerance decides whether a cached action
    # chunk survives the safety clamp at all.
    if max_target_step_rad is not None:
        command += ["--max-target-step-rad", str(max_target_step_rad)]
    if chunk_invalidation_tolerance_rad is not None:
        command += ["--chunk-invalidation-tolerance-rad",
                    str(chunk_invalidation_tolerance_rad)]
    if policy:
        command += ["--policy-cmd", policy]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    env.setdefault("TMPDIR", str(Path.home() / "tmp"))
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    if env_extra:
        env.update(env_extra)

    log_path = out_dir / "logs" / f"gpu{gpu}.log"
    print(f"[gpu{gpu}] {len(indices)} cases -> {log_path}", flush=True)
    began = time.time()
    with log_path.open("ab") as log:
        result = subprocess.run(
            command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    elapsed = time.time() - began

    # The artifact is the contract, not the return code: Kit's shutdown can turn
    # a Python traceback into exit 0.
    if result.returncode != 0 or not summary.is_file():
        with lock:
            failures.append({
                "gpu": gpu, "case_indices": indices,
                "returncode": result.returncode, "log": str(log_path),
                "elapsed_s": round(elapsed, 1),
            })
        print(f"[gpu{gpu}] FAILED rc={result.returncode} after {elapsed:.0f}s "
              f"-- see {log_path}", flush=True)
        return
    print(f"[gpu{gpu}] done in {elapsed:.0f}s", flush=True)


# --- helpers ---------------------------------------------------------------
def _resolve_suite(suite: str | Path) -> Path:
    path = Path(suite)
    if path.suffix == ".yaml" and path.is_file():
        return path.resolve()
    candidate = SUITES_DIR / f"{suite}.yaml"
    if candidate.is_file():
        return candidate.resolve()
    available = sorted(p.stem for p in SUITES_DIR.glob("*.yaml"))
    raise FileNotFoundError(f"unknown suite {suite!r}; available: {available}")


def _suite_interprior_root(suite_path: Path) -> str | None:
    """The env checkout this suite declares, or None if it does not.

    Read straight from the yaml rather than through `load_suite`, so resolving
    the root never depends on the case list being loadable.
    """

    import yaml

    payload = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
    declared = payload.get("interprior_root")
    return str(declared) if declared else None


def _count_cases(suite_path: Path, cases: int | None) -> int:
    from .suites.suite import load_suite

    loaded = load_suite(suite_path)
    if cases is not None:
        loaded.limit = cases
    return len(loaded.resolved_cases())


def _split(total: int, buckets: int) -> list[list[int]]:
    """Round-robin so a slow case does not land all its neighbours on one GPU."""

    out: list[list[int]] = [[] for _ in range(buckets)]
    for index in range(total):
        out[index % buckets].append(index)
    return out


def _default_out(suite_path: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path.home() / "bench_runs" / f"{suite_path.stem}_{stamp}"


def _merge(
    out_dir: Path,
    suite_path: Path,
    total: int,
    failures: list[dict[str, Any]],
    elapsed: float,
) -> Report:
    """Combine shard summaries. Counts add; fractions weight by case count."""

    summaries = []
    for path in sorted(out_dir.glob("gpu*/summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))

    report = Report(
        suite=suite_path.stem, cases=total, frames=0,
        mode="teacher_replay", out_dir=out_dir,
        failures=failures, elapsed_s=round(elapsed, 1),
    )
    if not summaries:
        return report

    report.frames = int(summaries[0].get("frames", 0))
    report.mode = str(summaries[0].get("mode", "teacher_replay"))
    report.physics_profile_sha256 = str(summaries[0].get("physics_profile_sha256", ""))
    report.physics_gate_ok = all(bool(s.get("physics_gate_ok")) for s in summaries)

    evaluated = 0
    weighted: dict[str, float] = {}
    for summary in summaries:
        count = int(summary.get("cases", 0))
        evaluated += count
        for key, value in (summary.get("v10_counts") or {}).items():
            if hasattr(report, key):
                setattr(report, key, getattr(report, key) + int(value))
        for key in ("guide_tracking_mean_m", "guide_tracking_fraction_within_3cm",
                    "guide_tracking_fraction_within_5cm"):
            if summary.get(key) is not None:
                weighted[key] = weighted.get(key, 0.0) + float(summary[key]) * count
        for name, value in (summary.get("class_counts") or {}).items():
            report.class_counts[name] = report.class_counts.get(name, 0) + int(value)
        report.cases_detail.extend(summary.get("cases_detail") or [])
        report.html_pages.extend(summary.get("html_pages") or [])
        report.shards.append({
            "case_indices": summary.get("case_indices"),
            "cases": count,
            "gate_pass": summary.get("gate_pass"),
        })

    if evaluated:
        for key, accumulated in weighted.items():
            setattr(report, key, accumulated / evaluated)
        medians = [s["pos_err_median_mm"] for s in summaries
                   if s.get("pos_err_median_mm") is not None]
        if medians:
            report.pos_err_median_mm = sum(medians) / len(medians)
    return report


def _report_dict(report: Report) -> dict[str, Any]:
    from dataclasses import asdict

    payload = asdict(report)
    payload["out_dir"] = str(report.out_dir)
    payload["ok"] = report.ok
    return payload


__all__ = [
    "run",
    "Report",
    "DEFAULT_ISAAC_PYTHON",
    "DEFAULT_INTERPRIOR_ROOT",
    "resolve_isaac_python",
    "resolve_interprior_root",
    "BUILD_HINT",
]
