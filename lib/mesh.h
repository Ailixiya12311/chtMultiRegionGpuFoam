// Device copy of the global multi-region mesh (see MeshIn in api.h) plus the cell -> face adjacency used by all
// gather-type kernels (deterministic sums, no atomics).
#pragma once

#include "par.h"
#include "vec.h"
#include "api.h"

namespace gc
{

// Cell-face adjacency entry: e >= 0 -> owner side of face e (e < nI: internal face, else boundary face e - nI);
// e < 0 -> neighbour side of internal face -e - 1.
HD inline bool cfOwner(int e) { return e >= 0; }
HD inline int cfFace(int e) { return e >= 0 ? e : -e - 1; }

struct DMesh
{
    int nC = 0, nI = 0, nB = 0, nIface = 0, nIr = 0;   // nIr: region internal faces (interface faces follow)

    dvec<int> own, nei;
    dvec<double> V, C, Sf, Cf, magSf;
    dvec<double> w, nonOrthDC, corrVec, lsP, lsN;
    dvec<double> dcIA, dcIB;
    dvec<double> dcB, lsB;
    dvec<int> cfPtr, cfEnt;
    dvec<int> cellRegion;

    std::vector<MeshIn::Region> regions;
    std::vector<MeshIn::Patch> patches;
    std::vector<MeshIn::Pair> pairs;

    void build(const MeshIn& m);
};

// raw-pointer view for kernels
struct MeshView
{
    int nC, nI, nB, nIface, nIr;
    const int *own, *nei, *cfPtr, *cfEnt, *cellRegion;
    const double *V, *C, *Sf, *Cf, *magSf, *w, *nonOrthDC, *corrVec, *lsP, *lsN, *dcIA, *dcIB, *dcB, *lsB;
};

MeshView view(const DMesh& m);

} // namespace gc
