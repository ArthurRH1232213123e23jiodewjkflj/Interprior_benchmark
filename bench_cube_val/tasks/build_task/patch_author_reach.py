#!/usr/bin/env python
import os
for f in ["reach_traj_author.html", "reach_traj_author.html.bak"]:
    if not os.path.exists(f):
        print("missing", f); continue
    s = open(f).read()
    o = s
    # set both reach values to 1.2 (any prior value)
    import re
    s = re.sub(r'"reach_max": [0-9.]+', '"reach_max": 1.2', s)
    s = re.sub(r'"reach_safe": [0-9.]+', '"reach_safe": 1.2', s)
    # drop the amber (safe) ring line -> keep only the hard-limit ring, and drop the dome
    s = s.replace('root.add(ring(RSAFE,TZ+0.001,0xd29922));  // comfortable on table\n', '')
    s = s.replace('root.add(ring(RSAFE,TZ+0.001,0xd29922));', '')
    if s != o:
        open(f, "w").write(s); print("patched", f)
    else:
        print("no-change", f)
