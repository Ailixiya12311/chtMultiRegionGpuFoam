"""Patch values of a fluid region in two results: face-area-weighted velocity, mass flux sum(rho U.Sf), T.

python diag_patch.py CASE_A CASE_B TIME REGION PATCH
"""
import os
import sys

sys.path.insert(0, "/py")
import numpy as np  # noqa: E402
from cht.foam.dictio import read_dict  # noqa: E402
from cht.mesh import Mesh  # noqa: E402

a, b, t, r, pn = sys.argv[1:6]
m = Mesh.read_dir(os.path.join(a, "constant", r, "polyMesh"), zones=False, log=lambda *x: None)
pt = next(p for p in m.patches if p["name"] == pn)
s0, n = pt["startFace"], pt["nFaces"]
Sf = np.asarray(m.Sf)[s0:s0 + n]
A = np.linalg.norm(Sf, axis=1)
print(f"{r}/{pn}: {n} faces, area {A.sum():.5f} m2")


def patch(case, fld, ncomp):
    d = read_dict(os.path.join(case, t, r, fld))
    e = d.sub("boundaryField").sub(pn)
    v = e.tokens("value")
    if v[0] == "uniform":
        return np.tile(np.asarray(v[1], float), (n, 1)) if ncomp == 3 else np.full(n, float(v[1]))
    return np.asarray(v[1].a, float)


for case in (a, b):
    U = patch(case, "U", 3)
    T = patch(case, "T", 1)
    try:
        rho = patch(case, "rho", 1)
    except Exception:
        rho = np.full(n, np.nan)
    mdot = -(rho*np.einsum("ij,ij->i", U, Sf)).sum()
    print(f"  {case}: |U| face mean {np.linalg.norm(U, axis=1).mean():.4f} m/s, max {np.linalg.norm(U, axis=1).max():.4f};"
          f" inflow sum(rho U.Sf) {mdot:.6f} kg/s; T mean {np.average(T, weights=A):.3f} K; rho mean {np.nanmean(rho):.5f}")
