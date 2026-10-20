/*---------------------------------------------------------------------------*\
    OpenFOAM side of the fluid regions (chtMultiRegionFoam v2606 fluid/createFluidFields.H, UEqn.H, pEqn.H,
    solveFluid.H): U, p_rgh, p, rho, phi read and written by OpenFOAM, held on the device during the run.

    Laminar, single-component, heRhoThermo with psi = 0 (Boussinesq / rhoConst), see FluidProps.

    Boundary slots of a fluid region: its boundary faces (global boundary order), then the faces of its
    region interfaces (walls for the flow), patch by patch.

    U:     fixedValue / noSlip (time-independent values), zeroGradient, inletOutlet, pressureInletOutletVelocity;
           assignable() and fixesValue() are OpenFOAM's own answers for each patch field
    p_rgh: fixedFluxPressure, zeroGradient, fixedValue (time-independent)
    Anything else stops with an error.
\*---------------------------------------------------------------------------*/

#ifndef gpuChtFluidCoupling_H
#define gpuChtFluidCoupling_H

#include "energyCoupling.H"
#include "surfaceFields.H"
#include "uniformDimensionedFields.H"
#include <map>

namespace Foam
{
namespace gpuCht
{

struct FlowPatchF1
{
    int fpIndex = -1;
    label patch = -1;
    bool hvacControlled = false;
    scalar hvacFlowRate = 0;
    autoPtr<Function1<scalar>> q;
};

struct VecPatchF1
{
    label s0 = 0, n = 0;
    autoPtr<Function1<vector>> f1;
};

struct ScaPatchF1
{
    label s0 = 0, n = 0;
    label specie = -1;
    autoPtr<Function1<scalar>> f1;
};

struct FluidRegion
{
    label r = -1;                  // region index
    int id = -1;                   // engine fluid id
    label nBr = 0, nS = 0;         // region boundary faces, slots
    labelList slot0;               // per patch: first slot (-1: empty patch)
    autoPtr<volVectorField> U;
    autoPtr<volScalarField> p_rgh, p, rho;
    autoPtr<surfaceScalarField> phi;
    PtrList<volScalarField> Y;
    autoPtr<volScalarField> k, epsilon, nut, alphat;
    bool turbulent = false;
    PtrList<FlowPatchF1> uFlowF1;
    PtrList<VecPatchF1> uValF1;
    PtrList<ScaPatchF1> pValF1, yValF1;
    // PIMPLE controls of the region
    bool momentumPredictor = true, frozenFlow = false;
    label nCorr = 1, nNonOrth = 0;
    scalar cumulativeContErr = 0;
    std::map<std::string, double> lastRes;
};

class FluidCoupling
{
public:
    FluidCoupling(Time& runTime, List<Region>& regions, const gc::MeshIn& mi, gc::Engine& E,
                  EnergyCoupling& energy, label nOuterCorr);

    label size() const { return F_.size(); }

    // start of a time step: rho.oldTime() for the energy (call before engine.energyStoreOld()), then the old
    // fluid levels and rhoEqn
    void rhoToEnergy();
    void storeOld(scalar dt);
    void hvacControl(scalar t0, scalar dt);
    scalar courantNo(scalar dt, scalar& meanCo) const;
    // solveFluid.H parts of one outer corrector
    void momentum(scalar dt, bool finalIter);
    void species(scalar dt, bool finalIter);
    void toEnergy(scalar dt);
    void pressure(scalar dt, bool finalIter);
    void turbulence(scalar dt, bool finalIter);
    bool outerConverged(const dictionary& ocrc, scalar hRes) const;

    // device -> OpenFOAM fields before runTime.write()
    void writeBack();

private:
    Time& runTime_;
    List<Region>& R_;
    const gc::MeshIn& mi_;
    gc::Engine& E_;
    EnergyCoupling& energy_;
    label nOuterCorr_;
    PtrList<FluidRegion> F_;
    bool hasHvac_ = false;
    label hvacFluidIdx_ = -1;
    label hvacVapIdx_ = -1, hvacDryIdx_ = -1;

    void setup(FluidRegion& f);
    void setupHvac();
    void updateBCs(FluidRegion& f);
};

} // namespace gpuCht
} // namespace Foam

#endif
