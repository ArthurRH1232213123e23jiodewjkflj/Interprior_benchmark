#!/usr/bin/env python3
"""All test_3.7 real-grasp rollouts overlaid in ONE 3D coordinate system.
Each episode's cube follow-trajectory is one colored line; handoff = circle,
final = diamond. The shared goal path (slot11 air goals) is drawn once in red,
final goal as a red X. Grasp-phase (pre-handoff) cube path drawn faint grey.
"""
import os, glob, json
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

HERE = os.path.dirname(os.path.abspath(__file__))
CSVDIR = os.path.join(HERE, "csv") if os.path.isdir(os.path.join(HERE, "csv")) else HERE
ev = json.load(open(os.path.join(HERE, "eval.json"))) if os.path.exists(os.path.join(HERE,"eval.json")) else None
meta = {}
if ev:
    meta = {(e["env_id"], e["episode_index"]): e for e in ev["episodes"]}

CUBE = slice(27,30); GOAL = slice(34,37)
files = sorted(glob.glob(os.path.join(CSVDIR, "rollout_*.csv")))
palette = ["#58a6ff","#3fb950","#d29922","#f778ba","#a371f7","#ff7b72","#39c5cf"]

def uniq(g):
    pts,last=[],None
    for r in g:
        if np.any(np.isnan(r)): continue
        if last is None or np.linalg.norm(r-last)>1e-4: pts.append(r); last=r
    return np.asarray(pts) if pts else np.zeros((0,3))

fig = go.Figure()
goal_drawn = False
for i,f in enumerate(files):
    a = np.loadtxt(f, delimiter=",", dtype=np.float32)
    cube = a[:, CUBE]; goals = uniq(a[:, GOAL])
    valid = ~np.isnan(a[:, GOAL.start]); h = int(np.argmax(valid)) if valid.any() else 0
    base = os.path.basename(f).replace("rollout_","").replace(".csv","")
    env_id = int(base.split("_")[0].replace("env","")); ep = int(base.split("_ep")[1])
    m = meta.get((env_id,ep),{}); col = palette[i % len(palette)]
    lbl = f"{base} ({m.get('final_pos_err_m',0)*100:.0f}cm/{m.get('final_rot_err_deg',0):.0f}°)"
    # grasp phase faint
    fig.add_trace(go.Scatter3d(x=cube[:h,0],y=cube[:h,1],z=cube[:h,2],mode="lines",
        line=dict(color="rgba(120,120,120,0.25)",width=2),showlegend=False,hoverinfo="skip"))
    # follow phase colored
    fig.add_trace(go.Scatter3d(x=cube[h:,0],y=cube[h:,1],z=cube[h:,2],mode="lines",
        line=dict(color=col,width=5),name=lbl))
    # handoff point
    fig.add_trace(go.Scatter3d(x=[cube[h,0]],y=[cube[h,1]],z=[cube[h,2]],mode="markers",
        marker=dict(size=5,color=col,symbol="circle"),showlegend=False,hoverinfo="skip"))
    # final point
    fig.add_trace(go.Scatter3d(x=[cube[-1,0]],y=[cube[-1,1]],z=[cube[-1,2]],mode="markers",
        marker=dict(size=6,color=col,symbol="diamond"),showlegend=False,hoverinfo="skip"))
    # goal path once (shared slot11 sequence)
    if not goal_drawn and len(goals):
        fig.add_trace(go.Scatter3d(x=goals[:,0],y=goals[:,1],z=goals[:,2],mode="lines+markers",
            line=dict(color="crimson",width=5,dash="dash"),marker=dict(size=4,color="crimson"),
            name="goal path (slot11)"))
        fig.add_trace(go.Scatter3d(x=[goals[-1,0]],y=[goals[-1,1]],z=[goals[-1,2]],mode="markers",
            marker=dict(size=9,color="red",symbol="x"),name="final goal"))
        goal_drawn = True

n = len(files)
fig.update_layout(
    title=f"test_3.7 real-grasp — {n} rollouts in one coordinate (cube follow paths vs shared goal)",
    scene=dict(xaxis_title="x (m)",yaxis_title="y (m)",zaxis_title="z (m)",aspectmode="data"),
    height=800, width=1100, legend=dict(font=dict(size=11)))
out = os.path.join(HERE, "all_in_one_3.7.html")
fig.write_html(out, include_plotlyjs="cdn")
print(f"wrote {out} ({n} rollouts overlaid)")
