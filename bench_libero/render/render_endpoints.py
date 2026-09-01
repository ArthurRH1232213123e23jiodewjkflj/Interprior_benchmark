#!/usr/bin/env python3
"""All 67 lifted endpoints of test_3.7 real-grasp eval, placed relative to the
FIRST goal using the real measured quantities in eval.json:
  total_dist   = final_pos_err_m                         (known)
  vertical     = cube_zf_world_m - goal_z                (known)
  horizontal   = sqrt(total^2 - vertical^2)              (Pythagoras, real)
Azimuth is NOT stored, so we plot horizontal-distance (x) vs height-diff (y):
each endpoint sits at its true radial distance + true height relative to the
goal. Goal is the origin; a 3cm success ring is drawn. Color = dropped or held.
"""
import os, json
import numpy as np
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
ev = json.load(open(os.path.join(HERE, "eval.json")))
eps = ev["episodes"]
GOAL_Z = 0.63959   # first goal world/render z (constant across attempts, slot11)

tot = np.array([e["final_pos_err_m"] for e in eps])
zf  = np.array([e["cube_zf_world_m"] for e in eps])
dropped = np.array([e["cube_dropped_during_follow"] for e in eps])
vert = zf - GOAL_Z
horiz = np.sqrt(np.maximum(tot**2 - vert**2, 0.0))

fig = go.Figure()
# success ring (3cm) around goal at origin
th = np.linspace(0, 2*np.pi, 100)
fig.add_trace(go.Scatter(x=0.03*np.cos(th), y=0.03*np.sin(th), mode="lines",
    line=dict(color="#2ea043", width=2, dash="dot"), name="3cm success ring"))
# goal at origin
fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
    marker=dict(size=16, color="red", symbol="x"), name="first goal"))
# held endpoints
m = ~dropped
fig.add_trace(go.Scatter(x=horiz[m], y=vert[m], mode="markers",
    marker=dict(size=8, color="#58a6ff", opacity=0.75, line=dict(width=0.5,color="#0d1117")),
    name=f"held endpoint ({int(m.sum())})",
    text=[f"{d*100:.0f}cm" for d in tot[m]], hovertemplate="dist %{text}<extra></extra>"))
# dropped endpoints
d = dropped
fig.add_trace(go.Scatter(x=horiz[d], y=vert[d], mode="markers",
    marker=dict(size=9, color="#f85149", opacity=0.8, symbol="triangle-down"),
    name=f"dropped ({int(d.sum())})",
    text=[f"{x*100:.0f}cm" for x in tot[d]], hovertemplate="dist %{text}<extra></extra>"))

fig.update_layout(
    title=(f"test_3.7 real-grasp — all {len(eps)} lifted endpoints relative to first goal "
           f"(median {np.median(tot)*100:.0f}cm away, 0 inside 3cm)"),
    xaxis_title="horizontal distance to goal (m)",
    yaxis_title="height above/below goal (m)  [+ = cube higher]",
    height=760, width=1000, plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
    font=dict(color="#c9d1d9"), legend=dict(font=dict(size=12)))
fig.update_xaxes(zeroline=True, zerolinecolor="#30363d", gridcolor="#21262d", scaleanchor="y", scaleratio=1)
fig.update_yaxes(zeroline=True, zerolinecolor="#30363d", gridcolor="#21262d")
out = os.path.join(HERE, "endpoints_3.7.html")
fig.write_html(out, include_plotlyjs="cdn")
print(f"wrote {out}: {len(eps)} endpoints, held {int(m.sum())} / dropped {int(d.sum())}, "
      f"none within 3cm ({int((tot<0.03).sum())})")
