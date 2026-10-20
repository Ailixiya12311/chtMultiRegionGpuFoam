"""Extreme cells of one result: lowest / highest T and highest |U| of a fluid region, with centre, cell volume
and the patches the cell touches.

python diag_extreme.py CASE TIME [REGION] [N]
"""
import os
import sys

sys.path.insert(0, "/py")
sys.path.insert(0, "/py/verify")
import numpy as np  # noqa: E402
from compare_of import internal  # noqa: E402
from cht.mesh import Mesh  # noqa: E402

case, t = sys.argv[1], sys.argv[2]
r = sys.argv[3] if len(sys.argv) > 3 else "air"
n = int(sys.argv[4]) if len(sys.argv) > 4 else 8
m = Mesh.read_dir(os.path.join(case, "constant", r, "polyMesh"), zones=False, log=lambda *x: None)
C, V, own = np.asarray(m.C), np.asarray(m.V), np.asarray(m.own)
patch_of = {}
for pt in m.patches:
    for f in range(pt["startFace"], pt["startFace"] + pt["nFaces"]):
        patch_of.setdefault(int(own[f]), set()).add(pt["name"])
T = internal(os.path.join(case, t, r, "T"), len(C))
U = np.linalg.norm(internal(os.path.join(case, t, r, "U"), len(C)), axis=1) if r == "air" else None
print(f"{r}: {len(C)} cells, V min {V.min():.3e} median {np.median(V):.3e} max {V.max():.3e} m3")
def show(title, idx, val):
    print(title)
    for c in idx:
        print(f"   cell {c:7d} at {np.array2string(C[c], precision=3)}  {val[c]:10.4f}  V {V[c]:.2e}"
              f"  {sorted(patch_of.get(int(c), []))}")
show("lowest T", np.argsort(T)[:n], T)
show("highest T", np.argsort(-T)[:n], T)
if U is not None:
    show("highest |U|", np.argsort(-U)[:n], U)
