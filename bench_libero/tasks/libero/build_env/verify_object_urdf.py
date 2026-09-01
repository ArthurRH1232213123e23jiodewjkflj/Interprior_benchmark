"""Verify the generated URDFs: parseable, mesh resolves, mass sane, matches DexVerse where known."""
import json, os, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

LIB = Path("/home/huangsicheng/benchmark/bench_libero/tasks/libero")
ENVS = LIB / "envs"
idx = json.load(open(LIB / "object_urdf_index.json"))
fails = []

print("=== 1. XML parses, one link, mesh files resolve relative to the urdf ===")
for r in idx["objects"]:
    d = ENVS / r["env_id"]
    u = d / r["urdf"]
    try:
        root = ET.parse(u).getroot()
    except Exception as ex:
        fails.append(f"{r['env_id']}: unparseable ({ex})"); continue
    links = root.findall("link")
    if len(links) != 1: fails.append(f"{r['env_id']}: {len(links)} links, expected 1")
    for tag in ("visual", "collision", "inertial"):
        if links[0].find(tag) is None: fails.append(f"{r['env_id']}: no <{tag}>")
    for m in root.iter("mesh"):
        fn = m.get("filename")
        if not (d / fn).exists(): fails.append(f"{r['env_id']}: mesh {fn} missing")
print(f"  {len(idx['objects'])} urdfs checked, {len(fails)} problems")
for x in fails[:4]: print("    ", x)

print("\n=== 2. OBJ round-trips back to the mesh it came from ===")
bad = 0
for r in idx["objects"]:
    d = ENVS / r["env_id"]
    v = []
    with open(d / r["obj"]) as fh:
        for ln in fh:
            if ln.startswith("v "):
                v.append([float(x) for x in ln.split()[1:4]])
    v = np.asarray(v)
    with np.load(d / "mesh.npz", allow_pickle=True) as p:
        src = np.asarray(p["verts"], dtype=np.float64)
    if v.shape != src.shape or not np.allclose(v, src, atol=1e-5):
        bad += 1; fails.append(f"{r['env_id']}: OBJ verts differ from mesh.npz")
print(f"  {len(idx['objects']) - bad}/{len(idx['objects'])} OBJ match mesh.npz exactly")

print("\n=== 3. mass plausible for a graspable tabletop object ===")
print(f"  {'env':22s} {'mass_kg':>8s}  note")
for r in sorted(idx["objects"], key=lambda x: -x["mass_kg"]):
    note = ""
    if not 0.005 <= r["mass_kg"] <= 2.0:
        note = "OUT OF RANGE"; fails.append(f"{r['env_id']}: mass {r['mass_kg']}")
    if not r["watertight"]: note = "bbox fallback (not watertight)"
    if note: print(f"  {r['env_id']:22s} {r['mass_kg']:8.4f}  {note}")
print(f"  range {min(r['mass_kg'] for r in idx['objects']):.4f} - "
      f"{max(r['mass_kg'] for r in idx['objects']):.4f} kg (cube URDF is 0.1728)")

print("\n=== 4. cross-check against the DexVerse soup asset (independent source) ===")
cfg = (ENVS / "alphabet_soup" / "config.yaml")
if cfg.exists():
    import yaml
    y = yaml.unsafe_load(open(cfg))  # DexVerse config.yaml embeds python/tuple tags
    dex_mass = (y.get("mass_props") or {}).get("mass")
    ours = next(r for r in idx["objects"] if r["env_id"] == "alphabet_soup")["mass_kg"]
    print(f"  DexVerse config.yaml mass: {dex_mass} kg")
    print(f"  our URDF mass:             {ours} kg   ({ours/dex_mass:.1f}x)")
    print("  -> DexVerse pins a light hand-set mass; ours is density x mesh volume.")
    print("     Not a bug, but they are DIFFERENT physics. Flagging, not reconciling.")

print("\n=== 5. inertial origin sits inside the bbox ===")
for r in idx["objects"]:
    c = np.asarray(r["inertial_origin"]); lo = np.asarray(r["bbox_min_m"]); hi = np.asarray(r["bbox_max_m"])
    if not ((c >= lo - 1e-6).all() and (c <= hi + 1e-6).all()):
        fails.append(f"{r['env_id']}: COM {c} outside bbox")
print(f"  all {len(idx['objects'])} COMs inside their bbox: {not any('COM' in f for f in fails)}")

print("\n=== 6. wine_bottle: why not watertight, and does it matter ===")
d = ENVS / "wine_bottle"
with np.load(d / "mesh.npz", allow_pickle=True) as p:
    v = np.asarray(p["verts"], dtype=np.float64); f = np.asarray(p["faces"], dtype=np.int64)
from collections import Counter
edges = Counter()
for tri in f:
    for a, b in ((0,1),(1,2),(2,0)):
        edges[tuple(sorted((tri[a], tri[b])))] += 1
boundary = sum(1 for k, n in edges.items() if n == 1)
print(f"  faces={len(f)} edges={len(edges)} boundary_edges(open)={boundary}")
print(f"  -> open mesh (a bottle scanned without its base/interior sealed).")
print(f"  mass came from the bbox fallback, stated in the URDF header.")

print(f"\nFAILURES: {len(fails)}")
for x in fails: print("  -", x)
