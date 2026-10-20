"""Where do two results of the same case differ most? Lists the cells with the largest |U_A - U_B| (and T) of one
fluid region with their centres, both values and the cell's faces on patches.

python diag_spike.py CASE_A CASE_B TIME [REGION] [N]
"""
import os
import sys

sys.path.insert(0, "/py")
sys.path.insert(0, "/py/verify")
import numpy as np  # noqa: E402
from compare_of import internal  # noqa: E402
from cht.mesh import Mesh  # noqa: E402

a, b, t = sys.argv[1], sys.argv[2], sys.argv[3]
r = sys.argv[4] if len(sys.argv) > 4 else "air"
n = int(sys.argv[5]) if len(sys.argv) > 5 else 12
m = Mesh.read_dir(os.path.join(a, "constant", r, "polyMesh"), zones=False, log=lambda *x: None)
C = np.asarray(m.C)
nC = len(C)
own = np.asarray(m.own)
patch_of = {}
for pt in m.patches:
    for f in range(pt["startFace"], pt["startFace"] + pt["nFaces"]):
        patch_of.setdefault(int(own[f]), set()).add(pt["name"])
lo, hi = C.min(0), C.max(0)
print(f"{r}: {nC} cells, box {lo} .. {hi}")
for fld in ("U", "T"):
    xa = internal(os.path.join(a, t, r, fld), nC)
    xb = internal(os.path.join(b, t, r, fld), nC)
    d = np.linalg.norm(xa - xb, axis=1) if xa.ndim == 2 else np.abs(xa - xb)
    va = np.linalg.norm(xa, axis=1) if xa.ndim == 2 else xa
    vb = np.linalg.norm(xb, axis=1) if xb.ndim == 2 else xb
    print(f"== {fld}: max |A| {np.abs(va).max():.4g}, max |B| {np.abs(vb).max():.4g}; "
          f"cells with diff > 10 x rms: {(d > 10 * np.sqrt((d**2).mean())).sum()}")
    for c in np.argsort(-d)[:n]:
        print(f"   cell {c:8d} at {np.array2string(C[c], precision=4)}  A {va[c]:.6g}  B {vb[c]:.6g}  diff {d[c]:.4g}"
              f"  patches {sorted(patch_of.get(int(c), []))}")
