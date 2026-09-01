"""Convert zjw's held-out test_selection.json into a benchmark case index.

The 200 episodes are already chosen: `hierarchical_angle20_radius5_quantile_hash_v1`
at seed 768500200, all success/completed, zero overlap with the 500 exp04 trained
on. We only remap field names -- `derived_shard` -> `shard`, `derived_slot` ->
`slot` -- and carry the provenance through so the report says where each case
came from. No re-selection, no filtering: changing the set would change what the
number means.
"""
import json
from pathlib import Path

SRC = Path(
    "/home/zhangjiawei/Interprior_train/runs/data_selections/"
    "lqr_success_train500_test200_first768_seed768500200_20260822/test_selection.json"
)
OUT = Path("/home/huangsicheng/benchmark/train_eval/suites/exp04_test200_cases.json")
EXPECT_SHA = "7b1fe52bec5c41377a6e9ed50aa2c7fc7036e2a95c08789b5e9bd03c56517f21"

import hashlib

raw = SRC.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
print(f"selection sha256 : {digest}")
print(f"benchmark expects: {EXPECT_SHA}")
print(f"MATCH            : {digest == EXPECT_SHA}")

payload = json.loads(raw)
episodes = payload["episodes"]
print(f"\nsplit={payload.get('split')!r} count={payload.get('selection_count')} "
      f"episodes={len(episodes)}")

cases = []
for index, episode in enumerate(episodes):
    status = episode.get("episode_status")
    collection = episode.get("collection_status")
    if status != "success" or collection != "completed":
        raise ValueError(f"episode {index} is {status}/{collection}, expected success/completed")
    cases.append({
        "shard": episode["derived_shard"],
        "slot": int(episode["derived_slot"]),
        "case_index": index,
        "episode_uid": episode.get("episode_uid"),
        "attempt_uid": episode.get("attempt_uid"),
        "episode_status": status,
        "outcome": status,
        "selection_rank": episode.get("cell_selection_rank"),
        "split": "test",
    })

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(cases, indent=1), encoding="utf-8")
print(f"\nwrote {len(cases)} cases -> {OUT}")
print(f"case0: {json.dumps(cases[0])[:220]}")
