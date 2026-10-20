"""Rewrite the initial p of a fluid region as hydrostatic, p = P0 + RHO (g.(x - hRef)), cells and patch faces, so
that chtMultiRegionFoam's start-up p_rgh = p - rho gh is uniform (= P0) instead of carrying a rho g gradient (a
uniform p file means p_rgh varies by rho g dz; with a large first time step the momentum predictor turns that
into metres per second).

python hydro_p.py TIMEDIR REGION P0 RHO        (TIMEDIR: the case's start time directory, e.g. case/43200)
g is read from constant/<REGION>/g or constant/g, hRef from constant/<REGION>/hRef (0 when missing).
"""
import os
import re
import sys

sys.path.insert(0, "/py")
import numpy as np  # noqa: E402
from cht.mesh import Mesh  # noqa: E402

tdir, r, P0, rho = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
case = os.path.dirname(os.path.abspath(tdir))


def value(path):
    s = open(path).read()
    return s[s.index("value", s.index("dimensions")):]


gf = os.path.join(case, "constant", r, "g")
gf = gf if os.path.exists(gf) else os.path.join(case, "constant", "g")
g = np.array([float(x) for x in re.search(r"value\s*\(([^)]*)\)", value(gf)).group(1).split()])
hf = os.path.join(case, "constant", r, "hRef")
hRef = float(re.search(r"value\s+([-+0-9.eE]+)", value(hf)).group(1)) if os.path.exists(hf) else 0.0
ghRef = -np.linalg.norm(g)*hRef
m = Mesh.read_dir(os.path.join(case, "constant", r, "polyMesh"), zones=False, log=lambda *x: None)
C, Cf = np.asarray(m.C), np.asarray(m.Cf)
pc = P0 + rho*(C @ g - ghRef)


def lst(v):
    return "nonuniform List<scalar> %d\n(\n%s\n)" % (len(v), "\n".join("%.10g" % x for x in v))


p = os.path.join(tdir, r, "p")
s = open(p).read()
s, n = re.subn(r"internalField\s+[^;]*;", lambda _: "internalField " + lst(pc) + ";", s, count=1)
assert n == 1
for pt in m.patches:
    a, b = pt["startFace"], pt["startFace"] + pt["nFaces"]
    if b == a:
        continue
    pf = P0 + rho*(Cf[a:b] @ g - ghRef)
    pat = r"(\n\s+" + re.escape(pt["name"]) + r"\s*\{[^}]*?value\s+)uniform\s+[-+0-9.eE]+;"
    s, n = re.subn(pat, lambda mm: mm.group(1) + lst(pf) + ";", s, count=1)
    if n != 1:
        sys.exit("hydro_p: patch %s: no 'value uniform' entry in %s" % (pt["name"], p))
open(p, "w").write(s)
print("hydro_p: %s/%s: p %.3f .. %.3f (rho %.5g, hRef %g)" % (tdir, r, pc.min(), pc.max(), rho, hRef))
