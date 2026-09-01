#!/usr/bin/env python3
"""ALL 67 lifted endpoints in 3D, world-oriented axes with the ROBOT BASE at the
origin. In parallel eval each env sits at its own world grid cell (metres apart),
so raw world coords can't be overlaid; we subtract each env's own base world pose
(cube_final_base). The base quaternion is identity (20260731), so this is a PURE
TRANSLATION — axes point the same way as world, only the origin moves to the base.
Rotatable 3D scatter; the 3cm success region is a real sphere (no 2D-projection
illusion). Uses driver-logged cube_final_base / goal_used_base for all 67 episodes.
"""
import os, json
import numpy as np
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
ev = json.load(open(os.path.join(HERE, "eval.json")))
eps = ev["episodes"]
if not all("cube_final_base" in e for e in eps):
    raise SystemExit("eval.json lacks cube_final_base; re-run the goalprobe driver.")

HELD_ONLY = os.environ.get("HELD_ONLY", "0") == "1"   # drop the 9 dropped -> 58 held
if HELD_ONLY:
    eps = [e for e in eps if not e.get("cube_dropped_during_follow", False)]
P = np.array([e["cube_final_base"] for e in eps])           # base-origin, world axes
dropped = np.array([e["cube_dropped_during_follow"] for e in eps])
G = np.array(eps[0]["goal_used_base"])
dist = np.linalg.norm(P - G[None, :], axis=1)

fig = go.Figure()
# 3cm success sphere around goal
u, v = np.mgrid[0:2*np.pi:24j, 0:np.pi:12j]; r = 0.03
fig.add_trace(go.Surface(
    x=G[0]+r*np.cos(u)*np.sin(v), y=G[1]+r*np.sin(u)*np.sin(v), z=G[2]+r*np.cos(v),
    opacity=0.25, showscale=False, colorscale=[[0, "#2ea043"], [1, "#2ea043"]],
    hoverinfo="skip", name="3cm ring"))
# robot base at origin
fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers+text",
    marker=dict(size=6, color="#e3b341", symbol="square"),
    text=["base"], textposition="bottom center", name="robot base (origin)"))
# first goal
fig.add_trace(go.Scatter3d(x=[G[0]], y=[G[1]], z=[G[2]], mode="markers",
    marker=dict(size=6, color="red", symbol="x"), name="first goal"))
# held / dropped endpoints (all 67)
m = ~dropped
fig.add_trace(go.Scatter3d(x=P[m, 0], y=P[m, 1], z=P[m, 2], mode="markers",
    marker=dict(size=4, color="#58a6ff", opacity=0.85),
    name=f"held ({int(m.sum())})",
    text=[f"{d*100:.0f}cm" for d in dist[m]], hovertemplate="%{text} to goal<extra></extra>"))
d = dropped
fig.add_trace(go.Scatter3d(x=P[d, 0], y=P[d, 1], z=P[d, 2], mode="markers",
    marker=dict(size=4, color="#f85149", opacity=0.85, symbol="diamond"),
    name=f"dropped ({int(d.sum())})",
    text=[f"{x*100:.0f}cm" for x in dist[d]], hovertemplate="%{text} to goal<extra></extra>"))

n_in = int((dist < 0.03).sum())
fig.update_layout(
    title=(f"ALL {len(eps)} lifted endpoints in 3D (robot base at origin, world axes) — "
           f"median {np.median(dist)*100:.0f}cm to first goal, {n_in} inside 3cm"),
    scene=dict(xaxis_title="x — forward from base (m)", yaxis_title="y — lateral (m)",
               zaxis_title="z — height (m)", aspectmode="data",
               xaxis=dict(backgroundcolor="#0d1117", gridcolor="#21262d"),
               yaxis=dict(backgroundcolor="#0d1117", gridcolor="#21262d"),
               zaxis=dict(backgroundcolor="#0d1117", gridcolor="#21262d")),
    paper_bgcolor="#0d1117", font=dict(color="#c9d1d9"), height=800, width=1000)
out = os.path.join(HERE, "endpoints_held58_3d.html" if HELD_ONLY else "endpoints_all67_3d.html")
fig.write_html(out, include_plotlyjs="cdn")
print(f"wrote {out}: {len(eps)} endpoints, held {int(m.sum())} / dropped {int(d.sum())}, "
      f"goal=({G[0]:.3f},{G[1]:.3f},{G[2]:.3f}), {n_in} inside 3cm")
