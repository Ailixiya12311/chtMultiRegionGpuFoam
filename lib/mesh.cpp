#include "mesh.h"

#include <cstdarg>

namespace gc
{

static void need(const std::vector<double>& v, long n, const char* what)
{
    if (long(v.size()) != n)
        fatal("MeshIn.%s has %ld entries, expected %ld", what, long(v.size()), n);
}

void DMesh::build(const MeshIn& m)
{
    nC = m.nC;
    nI = m.nI;
    nB = m.nB;
    nIface = m.nIface;
    nIr = nI - nIface;
    const long nF = long(nI) + nB;
    if (long(m.own.size()) != nF || long(m.nei.size()) != nI)
        fatal("MeshIn: owner/neighbour sizes %ld/%ld, expected %ld/%d", long(m.own.size()), long(m.nei.size()), nF, nI);
    need(m.V, nC, "V");
    need(m.C, 3L*nC, "C");
    need(m.Sf, 3*nF, "Sf");
    need(m.Cf, 3*nF, "Cf");
    need(m.w, nI, "w");
    need(m.nonOrthDC, nI, "nonOrthDC");
    need(m.corrVec, 3L*nI, "corrVec");
    need(m.lsP, 3L*nI, "lsP");
    need(m.lsN, 3L*nI, "lsN");
    need(m.dcI_A, nIface, "dcI_A");
    need(m.dcI_B, nIface, "dcI_B");
    need(m.dcB, nB, "dcB");
    need(m.lsB, 3L*nB, "lsB");

    own = to_dev(m.own);
    nei = to_dev(m.nei);
    V = to_dev(m.V);
    C = to_dev(m.C);
    Sf = to_dev(m.Sf);
    Cf = to_dev(m.Cf);
    w = to_dev(m.w);
    nonOrthDC = to_dev(m.nonOrthDC);
    corrVec = to_dev(m.corrVec);
    lsP = to_dev(m.lsP);
    lsN = to_dev(m.lsN);
    dcIA = to_dev(m.dcI_A);
    dcIB = to_dev(m.dcI_B);
    dcB = to_dev(m.dcB);
    lsB = to_dev(m.lsB);
    regions = m.regions;
    patches = m.patches;
    pairs = m.pairs;

    magSf.resize(nF);
    {
        double* ms = ptr(magSf);
        const double* sf = ptr(Sf);
        for_n(nF, [=] HD (long f) { ms[f] = mag(ld3(sf, f)); });
    }

    // cell -> faces adjacency (host build: face order within each cell, deterministic)
    std::vector<int> cnt(size_t(nC) + 1, 0);
    for (int f = 0; f < nI; ++f) { cnt[m.own[f] + 1]++; cnt[m.nei[f] + 1]++; }
    for (int b = 0; b < nB; ++b) cnt[m.own[nI + b] + 1]++;
    for (int c = 0; c < nC; ++c) cnt[c + 1] += cnt[c];
    std::vector<int> ent(static_cast<size_t>(cnt[nC]));
    std::vector<int> pos(cnt.begin(), cnt.end() - 1);
    for (int f = 0; f < nI; ++f)
    {
        ent[pos[m.own[f]]++] = f;
        ent[pos[m.nei[f]]++] = -f - 1;
    }
    for (int b = 0; b < nB; ++b) ent[pos[m.own[nI + b]]++] = nI + b;
    cfPtr = to_dev(cnt);
    cfEnt = to_dev(ent);

    std::vector<int> reg(size_t(nC), -1);
    for (size_t r = 0; r < regions.size(); ++r)
        for (int c = regions[r].c0; c < regions[r].c0 + regions[r].nC; ++c) reg[c] = int(r);
    for (int c = 0; c < nC; ++c)
        if (reg[c] < 0) fatal("MeshIn: cell %d belongs to no region", c);
    cellRegion = to_dev(reg);
}

MeshView view(const DMesh& m)
{
    MeshView v;
    v.nC = m.nC; v.nI = m.nI; v.nB = m.nB; v.nIface = m.nIface; v.nIr = m.nIr;
    v.own = ptr(m.own); v.nei = ptr(m.nei); v.cfPtr = ptr(m.cfPtr); v.cfEnt = ptr(m.cfEnt);
    v.cellRegion = ptr(m.cellRegion);
    v.V = ptr(m.V); v.C = ptr(m.C); v.Sf = ptr(m.Sf); v.Cf = ptr(m.Cf); v.magSf = ptr(m.magSf);
    v.w = ptr(m.w); v.nonOrthDC = ptr(m.nonOrthDC); v.corrVec = ptr(m.corrVec);
    v.lsP = ptr(m.lsP); v.lsN = ptr(m.lsN); v.dcIA = ptr(m.dcIA); v.dcIB = ptr(m.dcIB);
    v.dcB = ptr(m.dcB); v.lsB = ptr(m.lsB);
    return v;
}

} // namespace gc
