"""Cell-ordering locality of an OpenFOAM mesh (memory access pattern of cell-face loops): for the internal faces,
the distance |owner - neighbour| between the two cell indices, and OpenFOAM's band (largest distance).

python ordering.py POLYMESH_DIR [...]        (ascii or binary owner / neighbour, label=32)
"""
import os
import re
import sys

import numpy as np


def read_labels(path):
    raw = open(path, "rb").read()
    head = raw[:4096].decode("latin-1")
    binary = "format      binary" in head or re.search(r"format\s+binary", head) is not None
    body_start = raw.index(b"}", raw.index(b"FoamFile")) + 1
    m = re.compile(rb"\s*(?://[^\n]*\n\s*)*(\d+)\s*\(").match(raw, body_start)
    if m is None:
        raise RuntimeError(path + ": cannot find the list size")
    n = int(m.group(1))
    if binary:
        return np.frombuffer(raw, dtype=np.int32, count=n, offset=m.end())
    return np.array(raw[m.end():].split(b")")[0].split(), dtype=np.int64)


for d in sys.argv[1:]:
    own = read_labels(os.path.join(d, "owner"))
    nei = read_labels(os.path.join(d, "neighbour"))
    nI = len(nei)
    nC = int(max(own.max(), nei.max())) + 1
    dist = np.abs(own[:nI].astype(np.int64) - nei.astype(np.int64))
    print("%-70s cells %9d  internal faces %9d  |own-nei| mean %10.1f  median %8.0f  p90 %9.0f  band %9d  "
          "mean/cells %.2e" % (d[-70:], nC, nI, dist.mean(), np.median(dist), np.percentile(dist, 90), dist.max(),
                                dist.mean()/nC))
