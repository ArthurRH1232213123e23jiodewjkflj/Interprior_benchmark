#!/usr/bin/env python3
"""Build a browsable index over an eval campaign's per-case metrics.

Reads eval/<name>/case*/metrics.json plus summary.json and writes index.html:
a sortable table, one row per case, linking to each case's rollout.html when
one exists. Re-run it after viewers are generated and the links light up.
"""
import argparse
import glob
import html
import json
import os

COLS = [
    ("case", "case", "s"),
    ("rank", "selection_rank", "d"),
    ("status", "rollout_status", "s"),
    ("grasp", "grasp_proxy_success", "b"),
    ("lifted", "ever_lifted", "b"),
    ("held", "final_lifted", "b"),
    ("dropped", "dropped_after_lift", "b"),
    ("&lt;3cm", "guide_tracking_fraction_within_3cm", "f4"),
    ("&lt;5cm", "guide_tracking_fraction_within_5cm", "f4"),
    ("final err (m)", "final_guide_tracking_error_m", "f4"),
    ("arm MAE (rad)", "action_mae_arm_rad", "f3"),
    ("hand MAE (rad)", "action_mae_hand_rad", "f3"),
    ("grasp frames", "grasp_proxy_longest_frames", "d"),
    ("frames", "frames", "d"),
]


def fmt(val, kind):
    if val is None:
        return '<span class="na">null</span>'
    if kind == "b":
        cls = "yes" if val is True else "no"
        return f'<span class="{cls}">{"yes" if val is True else "no"}</span>'
    if kind == "d":
        return str(int(val))
    if kind.startswith("f"):
        return f"%.{int(kind[1:])}f" % float(val)
    return html.escape(str(val))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="eval/<name> directory")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    root = os.path.abspath(args.dir)
    name = os.path.basename(root)
    summary = {}
    spath = os.path.join(root, "summary.json")
    if os.path.isfile(spath):
        summary = json.load(open(spath))

    ranks = {}
    cpath = os.path.join(root, "cases.json")
    if os.path.isfile(cpath):
        for c in json.load(open(cpath)):
            ranks[int(c["case_index"])] = c.get("selection_rank")

    rows = []
    for p in sorted(glob.glob(os.path.join(root, "case*", "metrics.json"))):
        d = json.load(open(p))
        cdir = os.path.dirname(p)
        cname = os.path.basename(cdir)
        idx = int("".join(ch for ch in cname if ch.isdigit()) or -1)
        d["case"] = cname
        d["selection_rank"] = ranks.get(idx)
        viewer = os.path.join(cdir, "rollout.html")
        d["_viewer"] = os.path.relpath(viewer, root) if os.path.isfile(viewer) else None
        rows.append(d)
    write_html(root, name, args.title or name, summary, rows)
    have = sum(1 for r in rows if r["_viewer"])
    print(f"[index] {len(rows)} cases, {have} with viewers -> "
          f"{os.path.join(root, 'index.html')}")


def write_html(root, name, title, summary, rows):
    res = summary.get("results", {})
    ex = summary.get("execution", {})
    proto = summary.get("protocol", {})
    ck = summary.get("checkpoint", {})
    sel = summary.get("selection", {})

    def card(label, value, note=""):
        return (f'<div class="card"><div class="lab">{label}</div>'
                f'<div class="val">{value}</div>'
                f'<div class="note">{note}</div></div>')

    n = ex.get("cases_completed", len(rows))
    cards = "".join([
        card("cases", f"{n}", f"{ex.get('cases_failed', 0)} failed"),
        card("grasp", f"{res.get('grasp_proxy_success', 0)}/{n}",
             f"{100.0 * (res.get('grasp_proxy_success_rate') or 0):.1f}%"),
        card("ever lifted", f"{res.get('ever_lifted', 0)}/{n}",
             f"{100.0 * (res.get('ever_lifted_rate') or 0):.1f}%"),
        card("mean &lt;3cm", f"{res.get('mean_guide_tracking_fraction_within_3cm', 0):.4f}",
             "geometric, not tracking &mdash; see caveat"),
        card("mean final err",
             f"{res.get('mean_final_guide_tracking_error_m', 0):.4f} m", ""),
        card("step", f"{ck.get('step', '?')}",
             f"val {ck.get('val_loss', '?')}"),
    ])

    thead = "".join(f"<th data-k='{k}'>{lab}</th>" for lab, k, _ in COLS) + "<th>viewer</th>"
    body = []
    for r in rows:
        tds = []
        for _, key, kind in COLS:
            tds.append(f"<td>{fmt(r.get(key), kind)}</td>")
        v = r["_viewer"]
        tds.append(f'<td>{f"<a href=%s>rollout</a>" % html.escape(v) if v else "<span class=na>&mdash;</span>"}</td>')
        body.append("<tr>" + "".join(tds) + "</tr>")

    caveats = summary.get("metric_caveats") or []
    cav = "".join(f"<li>{html.escape(c)}</li>" for c in caveats)

    doc = f"""<!doctype html>
<meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1a1a1a}}
 h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#666;margin-bottom:18px;font-size:13px}}
 .cards{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px}}
 .card{{border:1px solid #e3e3e3;border-radius:8px;padding:10px 14px;min-width:120px;background:#fafafa}}
 .lab{{font-size:11px;text-transform:uppercase;color:#888;letter-spacing:.04em}}
 .val{{font-size:20px;font-weight:600;margin:2px 0}}
 .note{{font-size:11px;color:#888}}
 table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
 th,td{{border-bottom:1px solid #ececec;padding:5px 8px;text-align:left;white-space:nowrap}}
 th{{background:#f6f6f6;cursor:pointer;position:sticky;top:0;font-size:12px}}
 th:hover{{background:#ececec}}
 tr:hover td{{background:#fcfcf7}}
 .yes{{color:#0a7d28;font-weight:600}} .no{{color:#999}} .na{{color:#bbb}}
 .caveat{{border-left:3px solid #d9a600;background:#fffbf0;padding:10px 14px;margin:18px 0;font-size:13px}}
 .caveat ul{{margin:6px 0 0 18px;padding:0}}
 code{{background:#f2f2f2;padding:1px 4px;border-radius:3px;font-size:12px}}
</style>
<h1>{html.escape(title)}</h1>
<div class="sub">
 ckpt <code>{html.escape(str(ck.get('path','?')))}</code><br>
 split <b>{proto.get('split','?')}</b> &middot; plan_frames {proto.get('plan_frames','?')}
 &middot; replan {proto.get('replan_interval','?')} &middot; conditioning
 {proto.get('object_flow_conditioning','?')} &middot; sampled {sel.get('sample_count','?')}
 of {sel.get('population_episodes','?')} (seed {sel.get('seed','?')})
 &middot; {ex.get('elapsed_s','?')}s on GPUs {summary.get('platform',{}).get('node_gpus','?')}
</div>
<div class="cards">{cards}</div>
{f'<div class="caveat"><b>Reading these numbers</b><ul>{cav}</ul></div>' if cav else ''}
<table id="t"><thead><tr>{thead}</tr></thead><tbody>
{chr(10).join(body)}
</tbody></table>
<script>
const t=document.getElementById('t');
t.querySelectorAll('th').forEach((th,i)=>th.onclick=()=>{{
  const tb=t.tBodies[0], rows=[...tb.rows];
  const dir=th.dataset.d==='1'?-1:1; th.dataset.d=dir===1?'1':'';
  const num=v=>{{const f=parseFloat(v);return isNaN(f)?null:f;}};
  rows.sort((a,b)=>{{
    const x=a.cells[i].innerText.trim(), y=b.cells[i].innerText.trim();
    const nx=num(x), ny=num(y);
    if(nx!==null&&ny!==null) return (nx-ny)*dir;
    return x.localeCompare(y)*dir;
  }});
  rows.forEach(r=>tb.appendChild(r));
}});
</script>
"""
    open(os.path.join(root, "index.html"), "w").write(doc)


if __name__ == "__main__":
    main()
