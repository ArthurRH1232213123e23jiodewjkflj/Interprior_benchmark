#!/usr/bin/env python3
"""Render LSTM follow-failure rollouts (test_3.7) as interactive 3D html.

CSV columns (per eval_goal_follow_tro_grasp_batched.py:481):
  0..26  = joint pos (Lab order, 27)
  27..29 = cube world pos xyz
  30..33 = cube world quat (wxyz)
  34..36 = current follow-goal world pos xyz (nan during grasp phase)
  37..40 = current follow-goal world quat
"""
import json
import os
import glob
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(HERE, "csv")
OUT = os.path.join(HERE, "fail_examples.html")

# ---- eval.json metrics keyed by (env_id, episode_index) ----
with open(os.path.join(HERE, "eval.json")) as f:
    ev = json.load(f)
meta = {(e["env_id"], e["episode_index"]): e for e in ev["episodes"]}
summ = ev["summary"]

CUBE = slice(27, 30)
GOAL = slice(34, 37)


def load(path):
    a = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return a


def unique_goals(goal_xyz):
    """Collapse the per-step goal column to the ordered sequence of distinct
    waypoints (goal only changes when the pointer advances)."""
    pts, last = [], None
    for row in goal_xyz:
        if np.any(np.isnan(row)):
            continue
        if last is None or np.linalg.norm(row - last) > 1e-4:
            pts.append(row)
            last = row
    return np.asarray(pts) if pts else np.zeros((0, 3))


files = sorted(glob.glob(os.path.join(CSV_DIR, "rollout_*.csv")))
n = len(files)
cols = 2
rows = (n + cols - 1) // cols
fig = make_subplots(
    rows=rows, cols=cols,
    specs=[[{"type": "scene"}] * cols for _ in range(rows)],
    subplot_titles=[os.path.basename(f) for f in files],
)

for i, f in enumerate(files):
    r, c = i // cols + 1, i % cols + 1
    a = load(f)
    cube = a[:, CUBE]
    goals = unique_goals(a[:, GOAL])
    # find where policy control starts (first non-nan goal row)
    valid = ~np.isnan(a[:, GOAL.start])
    handoff = int(np.argmax(valid)) if valid.any() else len(a)

    base = os.path.basename(f).replace("rollout_", "").replace(".csv", "")
    env_id = int(base.split("_")[0].replace("env", ""))
    ep = int(base.split("_ep")[1])
    m = meta.get((env_id, ep), {})
    ferr = m.get("final_pos_err_m", float("nan"))
    frot = m.get("final_rot_err_deg", float("nan"))
    dropped = m.get("cube_dropped_during_follow", False)
    sub_title = (f"{base}: final_err={ferr*100:.1f}cm rot={frot:.0f}deg "
                 f"{'DROPPED' if dropped else ''}")
    fig.layout.annotations[i].text = sub_title
    fig.layout.annotations[i].font.size = 11

    sc = (r, c)
    # cube actual path (grasp phase grey, follow phase blue)
    fig.add_trace(go.Scatter3d(
        x=cube[:handoff, 0], y=cube[:handoff, 1], z=cube[:handoff, 2],
        mode="lines", line=dict(color="lightgray", width=3),
        name="grasp phase", showlegend=(i == 0)), row=r, col=c)
    fig.add_trace(go.Scatter3d(
        x=cube[handoff:, 0], y=cube[handoff:, 1], z=cube[handoff:, 2],
        mode="lines", line=dict(color="royalblue", width=5),
        name="cube (follow)", showlegend=(i == 0)), row=r, col=c)
    # goal waypoints (target the policy was supposed to chase)
    if len(goals):
        fig.add_trace(go.Scatter3d(
            x=goals[:, 0], y=goals[:, 1], z=goals[:, 2],
            mode="lines+markers",
            line=dict(color="crimson", width=4, dash="dash"),
            marker=dict(size=4, color="crimson"),
            name="goal path", showlegend=(i == 0)), row=r, col=c)
        # final goal (the one final_err is measured against)
        fig.add_trace(go.Scatter3d(
            x=[goals[-1, 0]], y=[goals[-1, 1]], z=[goals[-1, 2]],
            mode="markers", marker=dict(size=8, color="red", symbol="x"),
            name="final goal", showlegend=(i == 0)), row=r, col=c)
    # cube final position (where it actually ended)
    fig.add_trace(go.Scatter3d(
        x=[cube[-1, 0]], y=[cube[-1, 1]], z=[cube[-1, 2]],
        mode="markers", marker=dict(size=8, color="blue", symbol="diamond"),
        name="cube final", showlegend=(i == 0)), row=r, col=c)

title = (f"LSTM follow failures (test_3.7)  |  grasp {summ['grasp_success_rate']*100:.0f}% "
         f"({summ['n_grasp_lift_ok']}/{summ['n_total_attempts']}), "
         f"follow pos+rot {summ['n_reached_any_goal_posrot']}/{summ['n_total_attempts']}, "
         f"final_err median {summ['final_err_m']['median']*100:.0f}cm  |  "
         f"blue=cube actual, red=goal target")
fig.update_layout(title=dict(text=title, font=dict(size=13)), height=460 * rows, width=1200)
fig.write_html(OUT, include_plotlyjs="cdn")
print(f"wrote {OUT}")
print(f"rendered {n} rollouts")
