"""Filter the hand self-collision pairs the upstream adjacency map does not know.

WHY THIS EXISTS. `scene_utils._apply_self_collision_filters` mirrors Isaac Gym --
enable every self-collision, then mask the adjacent pairs -- by reading
`isaacgymenvs/tasks/simtoolreal/adjacent_links.py`. That file describes a
DIFFERENT ROBOT: its link names are `iiwa14_link_0..7` and
`left/right_{index,middle,ring,pinky,thumb}_{DP,MP,PP,MC,MCP_VL,CMC_VL}` -- a
Sharpa hand on a KUKA iiwa14. Not one xArm7 or Wuji link name appears in it.

The upstream loader skips links it cannot find, by design and without failing,
so on our robot the masking step runs and masks NOTHING. Measured, in the run
log of the first self-collision-enabled run:

    [scene_utils] self-collision: filtered 0 adjacent link pairs across 28
    robot bodies in xarm7_wuji_right_physics.usd; skipped 52 merged/absent links

So enabling self-collision (0908) without this leaves the hand fully unmasked.
PhysX auto-filters directly-jointed parent/child pairs, so the finger chains are
fine, but `right_palm_link` <-> `right_fingerN_link2` is a GRANDPARENT pair --
palm to link1 to link2 -- which is not auto-filtered. Those base knuckles then
push against the palm and the hand fights itself.

WHICH PAIRS. From the DexArm_Description README for this exact arm+hand
(github.com/VanadiumZ/DexArm_Description), which states the filtering must be
done environment-side and lists it: fingers 2,3,4,5 `link2` against the palm.
Finger 1 (the thumb) is deliberately NOT filtered -- its base is far enough from
the palm that filtering it would hide a real collision.

The README's separate `link7` <-> `right_palm_link` question does not arise on
this import: the two ARE one body here, because the palm is merged into link7.

WHY ON THE LIVE STAGE, NOT THE USD FILE. Upstream edits
`<robot>_physics.usd` during conversion, which happens inside
`super()._setup_scene()`; by the time any of our code runs, the robot is already
spawned from it. Authoring `FilteredPairsAPI` on the spawned prims instead is
equivalent -- PhysX parses the stage when the sim starts playing, which is after
env construction -- and it needs no write access to the shared checkout.

THIS FUNCTION RAISES IF IT FILTERS NOTHING. The whole reason the gap survived is
that the upstream version reports zero and carries on.
"""

from __future__ import annotations

# THE PALM IS NOT A BODY. `merge_fixed_joints=True` folds it away -- the import
# log says so in as many words:
#
#   link right_palm_link has body properties (mass, inertia, or collisions)
#   and is being merged into link7
#
# so the palm's collider belongs to `link7`, and naming the palm here matches
# nothing. (The first version of this file did exactly that and the guard at the
# bottom caught it, which is the only reason it is not still wrong. 28 bodies
# survive import: link_base, link1..7, and finger{1..5}_link{1..4} -- the five
# tips and the palm are the six that merged away.)
#
# The pair to filter is therefore link7 <-> finger{2,3,4,5}_link2, and it is the
# README's palm-vs-link2 rule unchanged: after the merge, fingerN_link1 hangs
# directly off link7, so link7 <-> fingerN_link1 is a parent/child pair PhysX
# auto-filters, while link7 <-> fingerN_link2 is one joint further out and is
# not. Thumb (finger1) stays unfiltered on purpose -- see the module docstring.
WUJI_FILTER_PAIRS: tuple[tuple[str, str], ...] = tuple(
    ("link7", f"right_finger{i}_link2") for i in (2, 3, 4, 5)
)


def apply(env, pairs=WUJI_FILTER_PAIRS, *, verbose: bool = True) -> dict:
    """Author FilteredPairsAPI for `pairs` on every env's robot. Returns a report."""

    from pxr import Usd, UsdPhysics

    stage = env.sim.stage if hasattr(env, "sim") else env.scene.stage

    # Group robot rigid bodies by the env prim that owns them, so a pair is
    # filtered once per env rather than across envs.
    by_env: dict[str, dict[str, object]] = {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        path = str(prim.GetPath())
        if "/Robot" not in path:
            continue
        key = path.split("/Robot")[0]
        by_env.setdefault(key, {})[prim.GetName()] = prim

    applied, missing = 0, set()
    for key, bodies in by_env.items():
        for a_name, b_name in pairs:
            a, b = bodies.get(a_name), bodies.get(b_name)
            if a is None or b is None:
                missing.add(a_name if a is None else b_name)
                continue
            rel = UsdPhysics.FilteredPairsAPI.Apply(a).CreateFilteredPairsRel()
            if b.GetPath() not in set(rel.GetTargets()):
                rel.AddTarget(b.GetPath())
                applied += 1

    report = {"applied": applied, "envs": len(by_env),
              "pairs_per_env": len(pairs), "missing": sorted(missing)}
    if verbose:
        print(f"[hand_self_collision] filtered {applied} pair(s) across "
              f"{len(by_env)} env(s)"
              + (f"; MISSING LINKS {sorted(missing)}" if missing else ""), flush=True)
    if applied == 0:
        raise SystemExit(
            "hand self-collision filtering matched nothing. The link names in "
            f"{[p for pair in pairs for p in pair]} are not in this robot, or the "
            "robot prims were not found. Refusing to run: this is exactly the "
            "silent no-op the upstream adjacency map already had."
        )
    return report
