"""Convert a frozen upstream selection JSON into this package's case schema.

The selection is the authority on WHICH episodes are evaluated; this only renames
fields. Nothing is sampled, filtered or reordered here, so the output is a pure
function of the input digest -- which is why the digest is copied into the output
and printed for the suite header.

Field mapping (upstream -> ours):
    derived_shard -> shard        episode -> slot
    start_frame, max_frames, episode_uid carry over unchanged.
`object_uid` / `object_id` / `object_source` are carried through because the
per-case asset pin needs them to join object_catalog.parquet.

Usage:
    python -m bench_multi_val.suites.convert_selection \
        --selection /home/zhangjiawei/Interprior_train/configs/evaluation/selections/<f>.json \
        --out bench_multi_val/suites/objaverse_val436_cases.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED = ("derived_shard", "episode")


def convert(selection: dict) -> list[dict]:
    cases = selection.get("cases")
    if not cases:
        raise ValueError("selection has no 'cases'")

    out = []
    for index, case in enumerate(cases):
        missing = [k for k in REQUIRED if k not in case]
        if missing:
            raise ValueError(f"case {index} lacks {missing}")
        shard = Path(str(case["derived_shard"]))
        if not shard.is_dir():
            raise FileNotFoundError(f"case {index} shard does not exist: {shard}")
        out.append(
            {
                "case_index": index,
                "shard": str(shard),
                "slot": int(case["episode"]),
                "start_frame": int(case.get("start_frame", 0)),
                "max_frames": int(case["max_frames"]) if case.get("max_frames") else None,
                "episode_uid": case.get("episode_uid"),
                "split": selection.get("split", "validation"),
                "object_uid": case.get("object_uid"),
                "object_id": case.get("object_id"),
                "object_source": case.get("object_source"),
                "selection_case_id": case.get("case_id"),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    path = args.selection.expanduser().resolve(strict=True)
    raw = path.read_bytes()
    selection = json.loads(raw.decode("utf-8"))
    cases = convert(selection)

    # Every case must agree on the window, or a single suite-level
    # start_frame/max_frames would be wrong for some of them.
    starts = {c["start_frame"] for c in cases}
    frames = {c["max_frames"] for c in cases}
    if len(starts) != 1 or len(frames) != 1:
        raise ValueError(
            f"selection mixes windows: start_frame={sorted(starts)} "
            f"max_frames={sorted(frames)}; the suite pins one value for all cases"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cases, indent=1) + "\n", encoding="utf-8")

    print(f"wrote {args.out}  cases={len(cases)}")
    print(f"  source          {path}")
    print(f"  source sha256   {hashlib.sha256(raw).hexdigest()}")
    print(f"  selection_sha   {selection.get('selection_sha256', '(absent)')}")
    print(f"  publish_sha     {selection.get('source_publish_commit_sha256', '(absent)')}")
    print(f"  partition_sha   {selection.get('source_partition_commit_sha256', '(absent)')}")
    print(f"  window          start_frame={starts.pop()} max_frames={frames.pop()}")
    print(f"  distinct objects {len({c['object_uid'] for c in cases})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
