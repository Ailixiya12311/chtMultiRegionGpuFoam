/*---------------------------------------------------------------------------*\
    Region meshes -> one global mesh (gc::MeshIn) for the compute library.

    Cells region by region (fluids first, then solids, as regionProperties lists them). Internal faces: each
    region's own internal faces, then the faces of each thermally coupled mapped patch pair (side A -> side B),
    paired through mappedPatchBase (the same map OpenFOAM's coupled boundary conditions use). Boundary faces:
    the other non-empty patches region by region. All geometric factors are OpenFOAM's (weights,
    nonOrthDeltaCoeffs, nonOrthCorrectionVectors, leastSquaresVectors, fvPatch::deltaCoeffs).
\*---------------------------------------------------------------------------*/

#ifndef gpuChtRegions_H
#define gpuChtRegions_H

#include "fvMesh.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "api.h"

namespace Foam
{
namespace gpuCht
{

struct PatchMap
{
    enum Kind { BOUNDARY, IFACE_A, IFACE_B, EMPTY };
    Kind kind = EMPTY;
    label b0 = -1;                 // BOUNDARY: global boundary start
    label pair = -1;               // IFACE_*: pair index
    labelList gface;               // IFACE_*: global internal face of each patch face
    labelList ifaceIdx;            // IFACE_*: interface index (gface - nIr) of each patch face
};

struct Region
{
    word name;
    bool fluid = false;
    fvMesh* mesh = nullptr;
    label index = -1;              // position in the global region list
    label c0 = 0, f0 = 0, b0 = 0;  // first global cell, region internal face, boundary face
    List<PatchMap> patches;
};

// fills mi and the patch maps of every region
void buildGlobalMesh(List<Region>& regions, gc::MeshIn& mi);

} // namespace gpuCht
} // namespace Foam

#endif
