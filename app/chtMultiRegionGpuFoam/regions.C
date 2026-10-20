#include "regions.H"

#include "emptyFvPatch.H"
#include "leastSquaresVectors.H"
#include "mappedPatchBase.H"

namespace Foam
{
namespace gpuCht
{

static inline void push3(std::vector<double>& v, const vector& x)
{
    v.push_back(x.x());
    v.push_back(x.y());
    v.push_back(x.z());
}

void buildGlobalMesh(List<Region>& R, gc::MeshIn& mi)
{
    // ---------------------------------------------------------------- cells
    label nC = 0;
    forAll(R, r)
    {
        R[r].index = r;
        R[r].c0 = nC;
        nC += R[r].mesh->nCells();
        R[r].patches.setSize(R[r].mesh->boundary().size());
    }
    mi.nC = nC;
    mi.V.reserve(nC);
    mi.C.reserve(3*nC);
    forAll(R, r)
    {
        const fvMesh& m = *R[r].mesh;
        forAll(m.V(), c)
        {
            mi.V.push_back(m.V()[c]);
            push3(mi.C, m.C()[c]);
        }
    }

    // ---------------------------------------------------------------- region internal faces
    label nI = 0;
    mi.regions.resize(R.size());
    forAll(R, r)
    {
        const fvMesh& m = *R[r].mesh;
        R[r].f0 = nI;
        const leastSquaresVectors& ls = leastSquaresVectors::New(m);
        const labelUList& own = m.owner();
        const labelUList& nei = m.neighbour();
        const scalarField& w = m.weights().primitiveField();
        const scalarField& dc = m.nonOrthDeltaCoeffs().primitiveField();
        const vectorField& cv = m.nonOrthCorrectionVectors().primitiveField();
        const vectorField& Sf = m.Sf().primitiveField();
        const vectorField& Cf = m.Cf().primitiveField();
        const vectorField& pV = ls.pVectors().primitiveField();
        const vectorField& nV = ls.nVectors().primitiveField();
        forAll(nei, f)
        {
            mi.own.push_back(own[f] + R[r].c0);
            mi.nei.push_back(nei[f] + R[r].c0);
            push3(mi.Sf, Sf[f]);
            push3(mi.Cf, Cf[f]);
            mi.w.push_back(w[f]);
            mi.nonOrthDC.push_back(dc[f]);
            push3(mi.corrVec, cv[f]);
            push3(mi.lsP, pV[f]);
            push3(mi.lsN, nV[f]);
        }
        nI += nei.size();
        gc::MeshIn::Region& g = mi.regions[r];
        g.name = R[r].name;
        g.fluid = R[r].fluid;
        g.c0 = R[r].c0;
        g.nC = m.nCells();
        g.f0 = R[r].f0;
        g.nI = nei.size();
    }
    const label nIr = nI;

    // ---------------------------------------------------------------- interface pairs
    HashTable<label> regionIndex;
    forAll(R, r) regionIndex.insert(R[r].name, r);
    struct Pr { label rA, pA, rB, pB; };
    DynamicList<Pr> pairs;
    forAll(R, r)
    {
        const polyBoundaryMesh& pbm = R[r].mesh->boundaryMesh();
        forAll(pbm, p)
        {
            if (!isA<mappedPatchBase>(pbm[p])) continue;
            const mappedPatchBase& mpb = refCast<const mappedPatchBase>(pbm[p]);
            const word& sr = mpb.sampleRegion();
            if (sr == R[r].name || !regionIndex.found(sr)) continue;
            const label rB = regionIndex[sr];
            const polyBoundaryMesh& pbmB = R[rB].mesh->boundaryMesh();
            const label pB = pbmB.findPatchID(mpb.samplePatch());
            if (pB < 0)
                FatalErrorInFunction << R[r].name << "/" << pbm[p].name() << ": samplePatch " << mpb.samplePatch()
                    << " not found in region " << sr << exit(FatalError);
            if (!isA<mappedPatchBase>(pbmB[pB])
             || refCast<const mappedPatchBase>(pbmB[pB]).sampleRegion() != R[r].name
             || refCast<const mappedPatchBase>(pbmB[pB]).samplePatch() != pbm[p].name())
                FatalErrorInFunction << sr << "/" << pbmB[pB].name() << " does not sample back "
                    << R[r].name << "/" << pbm[p].name() << exit(FatalError);
            if (pbm[p].size() != pbmB[pB].size())
                FatalErrorInFunction << "coupled patches " << R[r].name << "/" << pbm[p].name() << " (" << pbm[p].size()
                    << " faces) and " << sr << "/" << pbmB[pB].name() << " (" << pbmB[pB].size() << ") do not match"
                    << exit(FatalError);
            if (rB < r) continue;              // the pair is taken from its first region
            pairs.append(Pr{r, p, rB, pB});
        }
    }
    forAll(pairs, k)
    {
        const Pr& pr = pairs[k];
        const fvMesh& mA = *R[pr.rA].mesh;
        const fvMesh& mB = *R[pr.rB].mesh;
        const fvPatch& fA = mA.boundary()[pr.pA];
        const fvPatch& fB = mB.boundary()[pr.pB];
        const label n = fA.size();
        // face of patch B sampled by each face of patch A (the coupled boundary conditions' own map)
        const mappedPatchBase& mpb = refCast<const mappedPatchBase>(mA.boundaryMesh()[pr.pA]);
        labelList nbr(identity(fB.size()));
        mpb.distribute(nbr);
        // conformal check
        boolList used(fB.size(), false);
        scalar dCmax = 0, dSmax = 0;
        forAll(nbr, i)
        {
            const label j = nbr[i];
            if (j < 0 || j >= fB.size() || used[j])
                FatalErrorInFunction << "interface " << mA.name() << "/" << fA.name() << " <-> " << mB.name() << "/"
                    << fB.name() << ": mapping is not one-to-one" << exit(FatalError);
            used[j] = true;
            const scalar s = Foam::sqrt(fA.magSf()[i]);
            dCmax = max(dCmax, mag(fA.Cf()[i] - fB.Cf()[j])/s);
            dSmax = max(dSmax, mag(fA.Sf()[i] + fB.Sf()[j])/fA.magSf()[i]);
        }
        if (dCmax > 1e-6 || dSmax > 1e-6)
            FatalErrorInFunction << "interface " << mA.name() << "/" << fA.name() << " <-> " << mB.name() << "/"
                << fB.name() << " is not conformal (centre gap " << dCmax << ", normal mismatch " << dSmax << ")"
                << exit(FatalError);
        const leastSquaresVectors& lsA = leastSquaresVectors::New(mA);
        const leastSquaresVectors& lsB = leastSquaresVectors::New(mB);
        const vectorField& vA = lsA.pVectors().boundaryField()[pr.pA];
        const vectorField& vB = lsB.pVectors().boundaryField()[pr.pB];
        const labelUList& cA = fA.faceCells();
        const labelUList& cB = fB.faceCells();
        PatchMap& mapA = R[pr.rA].patches[pr.pA];
        PatchMap& mapB = R[pr.rB].patches[pr.pB];
        mapA.kind = PatchMap::IFACE_A;
        mapB.kind = PatchMap::IFACE_B;
        mapA.pair = mapB.pair = k;
        mapA.gface.setSize(n);
        mapA.ifaceIdx.setSize(n);
        mapB.gface.setSize(n);
        mapB.ifaceIdx.setSize(n);
        gc::MeshIn::Pair P;
        P.rA = pr.rA;
        P.rB = pr.rB;
        P.pA = fA.name();
        P.pB = fB.name();
        P.f0 = nI;
        P.n = n;
        mi.pairs.push_back(P);
        forAll(nbr, i)
        {
            const label j = nbr[i];
            mi.own.push_back(cA[i] + R[pr.rA].c0);
            mi.nei.push_back(cB[j] + R[pr.rB].c0);
            push3(mi.Sf, fA.Sf()[i]);
            push3(mi.Cf, fA.Cf()[i]);
            mi.w.push_back(0.5);
            mi.nonOrthDC.push_back(0);
            push3(mi.corrVec, Zero);
            push3(mi.lsP, vA[i]);
            push3(mi.lsN, vB[j]);
            mi.dcI_A.push_back(fA.deltaCoeffs()[i]);
            mi.dcI_B.push_back(fB.deltaCoeffs()[j]);
            mapA.gface[i] = nI;
            mapA.ifaceIdx[i] = nI - nIr;
            mapB.gface[j] = nI;
            mapB.ifaceIdx[j] = nI - nIr;
            ++nI;
        }
    }
    mi.nI = nI;
    mi.nIface = nI - nIr;

    // ---------------------------------------------------------------- boundary faces
    label nB = 0;
    forAll(R, r)
    {
        const fvMesh& m = *R[r].mesh;
        R[r].b0 = nB;
        mi.regions[r].b0 = nB;
        const leastSquaresVectors& ls = leastSquaresVectors::New(m);
        forAll(m.boundary(), p)
        {
            PatchMap& pm = R[r].patches[p];
            if (pm.kind == PatchMap::IFACE_A || pm.kind == PatchMap::IFACE_B) continue;
            const fvPatch& fp = m.boundary()[p];
            if (isA<emptyFvPatch>(fp))
            {
                pm.kind = PatchMap::EMPTY;
                continue;
            }
            if (fp.coupled())
                FatalErrorInFunction << m.name() << "/" << fp.name() << ": coupled patch type " << fp.type()
                    << " is not supported (serial runs without cyclics)" << exit(FatalError);
            pm.kind = PatchMap::BOUNDARY;
            pm.b0 = nB;
            gc::MeshIn::Patch P;
            P.region = R[r].name;
            P.name = fp.name();
            P.type = fp.type();
            P.b0 = nB;
            P.n = fp.size();
            mi.patches.push_back(P);
            const labelUList& fc = fp.faceCells();
            const vectorField& lb = ls.pVectors().boundaryField()[p];
            forAll(fp, i)
            {
                mi.own.push_back(fc[i] + R[r].c0);
                push3(mi.Sf, fp.Sf()[i]);
                push3(mi.Cf, fp.Cf()[i]);
                mi.dcB.push_back(fp.deltaCoeffs()[i]);
                push3(mi.lsB, lb[i]);
                ++nB;
            }
        }
        mi.regions[r].nB = nB - R[r].b0;
    }
    mi.nB = nB;
    Info<< "global mesh: " << mi.nC << " cells, " << mi.nI << " internal faces (" << mi.nIface << " on "
        << mi.pairs.size() << " region interfaces), " << mi.nB << " boundary faces, " << R.size() << " regions"
        << nl << endl;
}

} // namespace gpuCht
} // namespace Foam
