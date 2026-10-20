"""Faces of given cells of one region: neighbour (cell or patch), face area, non-orthogonality (angle between the
face normal and the owner-neighbour vector), the distance ratio |d.n| / |d| and, with TIME, the cell's U and T and
the face flux. For looking at the mesh where a run goes wrong.

python diag_cells.py CASE REGION CELL [CELL...] [--time T]
"""
import os
import sys

sys.path.insert(0, "/py")
sys.path.insert(0, "/py/verify")
import numpy as np  # noqa: E402
from compare_of import internal  # noqa: E402
from cht.mesh import Mesh  # noqa: E402

args = sys.argv[1:]
t = None
if "--time" in args:
    i = args.index("--time")
    t = args[i + 1]
    del args[i:i + 2]
case, r, cells = args[0], args[1], [int(c) for c in args[2:]]
m = Mesh.read_dir(os.path.join(case, "constant", r, "polyMesh"), zones=False, log=lambda *x: None)
own, nei = np.asarray(m.own), np.asarray(m.nei)
Sf, Cf, C, V = np.asarray(m.Sf), np.asarray(m.Cf), np.asarray(m.C), np.asarray(m.V)
pname = np.repeat([p["name"] for p in m.patches], [p["nFaces"] for p in m.patches])
U = T = None
if t is not None:
    U = internal(os.path.join(case, t, r, "U"), m.nC)
    T = internal(os.path.join(case, t, r, "T"), m.nC)
for c in cells:
    fs = np.where(own == c)[0]
    fn = np.where(nei == c)[0]
    head = f"cell {c} at {np.array2string(C[c], precision=4)}  V {V[c]:.3e}"
    if U is not None:
        head += f"  U {np.array2string(U[c], precision=3)}  T {T[c]:.3f}"
    print(head)
    for f in np.concatenate([fs, fn]):
        s = Sf[f] if own[f] == c else -Sf[f]
        a = np.linalg.norm(s)
        n = s/a
        if f < m.nI:
            o = nei[f] if own[f] == c else own[f]
            d = C[o] - C[c]
            what = f"cell {o:7d}"
        else:
            o = None
            d = Cf[f] - C[c]
            what = f"patch {pname[f - m.nI]}"
        cosang = np.dot(d, n)/np.linalg.norm(d)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        extra = ""
        if U is not None and o is not None:
            extra = f"  nbr |U| {np.linalg.norm(U[o]):.3g} T {T[o]:.2f}"
        print(f"   face {f:8d} {what:40s} A {a:.3e}  n {np.array2string(n, precision=2)}  |d| {np.linalg.norm(d):.3e}"
              f"  nonorth {ang:5.1f}{extra}")
