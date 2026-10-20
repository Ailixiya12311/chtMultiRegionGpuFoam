"""How many thin (sliver) cells a region has and how sharply cell size jumps across faces: per cell the thickness
V / (largest face area), per internal face the volume ratio of its two cells and the non-orthogonality. Prints
counts by threshold and the worst cells (centre, thickness, volume, neighbours' largest volume).

python mesh_slivers.py CASE [REGION] [N]
"""
import os
import sys

sys.path.insert(0, "/py")
import numpy as np  # noqa: E402
from cht.mesh import Mesh  # noqa: E402

case = sys.argv[1]
r = sys.argv[2] if len(sys.argv) > 2 else "air"
nshow = int(sys.argv[3]) if len(sys.argv) > 3 else 8
m = Mesh.read_dir(os.path.join(case, "constant", r, "polyMesh"), zones=False, log=lambda *x: None)
own, nei = np.asarray(m.own), np.asarray(m.nei)
Sf, C, V = np.asarray(m.Sf), np.asarray(m.C), np.asarray(m.V)
A = np.linalg.norm(Sf, axis=1)
nC, nI = m.nC, m.nI
Amax = np.zeros(nC)
np.maximum.at(Amax, own, A)
np.maximum.at(Amax, nei, A[:nI])
thick = V/Amax
ratio = np.maximum(V[own[:nI]], V[nei])/np.minimum(V[own[:nI]], V[nei])
d = C[nei] - C[own[:nI]]
cosang = np.einsum("ij,ij->i", d, Sf[:nI])/(np.linalg.norm(d, axis=1)*A[:nI])
nonorth = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
Vnb = np.zeros(nC)
np.maximum.at(Vnb, own[:nI], V[nei])
np.maximum.at(Vnb, nei, V[own[:nI]])
print(f"{r}: {nC} cells, {nI} internal faces; volume {V.min():.2e} .. {V.max():.2e} m3, median {np.median(V):.2e}")
for t in (0.001, 0.0025, 0.005, 0.01):
    s = thick < t
    print(f"  thickness < {t*1000:4.1f} mm: {s.sum():8d} cells ({100*s.mean():.2f} %), {V[s].sum():.4f} m3")
for q in (5, 10, 20, 50):
    s = ratio > q
    print(f"  faces with volume ratio > {q:3d}: {s.sum():8d} ({100*s.mean():.3f} %)")
for a in (40, 55, 65, 70):
    s = nonorth > a
    print(f"  faces non-orthogonal > {a} deg: {s.sum():8d} ({100*s.mean():.3f} %)")
print("  thinnest cells:")
for c in np.argsort(thick)[:nshow]:
    print(f"    cell {c:8d} at {np.array2string(C[c], precision=3)}  thickness {thick[c]*1000:7.3f} mm  V {V[c]:.2e}"
          f"  largest neighbour V {Vnb[c]:.2e}")
