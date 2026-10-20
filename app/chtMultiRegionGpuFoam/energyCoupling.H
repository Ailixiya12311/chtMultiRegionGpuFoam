/*---------------------------------------------------------------------------*\
    OpenFOAM side of the monolithic energy equation: region T fields (read and written by OpenFOAM), region
    thermophysical properties, T boundary conditions translated to the device kinds and updated each solve,
    schemes / solver / relaxation settings, and the device -> OpenFOAM copy at write time.

    Boundary conditions of T:
      fixedValue family (fixesValue, values depend on time only): evaluated by OpenFOAM on the host each solve
      zeroGradient / symmetry*: zero gradient;  fixedGradient: fixed gradient;  mixed: OpenFOAM mixed
      externalWallHeatFluxTemperature: OpenFOAM's updateCoeffs ported (power / flux / coefficient modes)
      turbulentTemperatureCoupledBaffleMixed on region interfaces: part of the matrix (series conductance,
      thicknessLayers / kappaLayers add a layer resistance)
    Anything else stops with an error.
\*---------------------------------------------------------------------------*/

#ifndef gpuChtEnergyCoupling_H
#define gpuChtEnergyCoupling_H

#include "regions.H"
#include "volFields.H"
#include "mixedFvPatchFields.H"
#include "Function1.H"
#include "PatchFunction1.H"

namespace Foam
{

class exteriorSurfaceFvPatchScalarField
:
    public mixedFvPatchScalarField
{
    scalar hc_ = 0;
    scalar eps_ = 0.9, alpha_ = 0.7;
    bool hasEps_ = false, hasAlpha_ = false;

public:
    TypeName("gpuCht::exteriorSurface");

    exteriorSurfaceFvPatchScalarField(const fvPatch&, const DimensionedField<scalar, volMesh>&);
    exteriorSurfaceFvPatchScalarField(const fvPatch&, const DimensionedField<scalar, volMesh>&, const dictionary&);
    exteriorSurfaceFvPatchScalarField
    (
        const exteriorSurfaceFvPatchScalarField&,
        const fvPatch&,
        const DimensionedField<scalar, volMesh>&,
        const fvPatchFieldMapper&
    );
    exteriorSurfaceFvPatchScalarField(const exteriorSurfaceFvPatchScalarField&);
    exteriorSurfaceFvPatchScalarField
    (
        const exteriorSurfaceFvPatchScalarField&,
        const DimensionedField<scalar, volMesh>&
    );

    virtual tmp<fvPatchScalarField> clone() const
    {
        return tmp<fvPatchScalarField>(new exteriorSurfaceFvPatchScalarField(*this));
    }
    virtual tmp<fvPatchScalarField> clone(const DimensionedField<scalar, volMesh>& iF) const
    {
        return tmp<fvPatchScalarField>(new exteriorSurfaceFvPatchScalarField(*this, iF));
    }

    scalar hc() const { return hc_; }
    bool hasEps() const { return hasEps_; }
    scalar eps() const { return eps_; }
    bool hasAlpha() const { return hasAlpha_; }
    scalar alpha() const { return alpha_; }

    virtual void write(Ostream&) const;
};

namespace gpuCht
{

struct TPatch
{
    enum Kind { HOSTFIXED, ZEROGRAD, FIXEDGRAD, MIXED, INLETOUTLET, EXTWALL, EXTSURF, COUPLED, EMPTY };
    Kind kind = EMPTY;
    label region = -1, patch = -1, b0 = -1, n = 0;
    word type;
    autoPtr<Function1<scalar>> inletVal;
    // externalWallHeatFluxTemperature
    word mode;
    autoPtr<Function1<scalar>> Q, Ta;
    autoPtr<PatchFunction1<scalar>> q, h;
    scalar totalSolidRes = 0, emissivity = 0, relaxation = 1;
};

struct RadSource
{
    int zone = -1;
    autoPtr<Function1<scalar>> Q;
};

struct SemiImplicitSource
{
    label region = -1;
    label specie = -1;                  // -1: energy (h); >= 0: specie index in FluidProps::species
    std::vector<int> cells;             // global cell indices (h) or region-local cell indices (specie)
    std::vector<double> wV;             // V_c / sum(V) (absolute) or V_c (specific)
    autoPtr<Function1<scalar>> SuF, SpF;
    scalar SuConst = 0, SpConst = 0;
};

struct FixedTempConstraint
{
    label region = -1;
    std::vector<int> gcells;            // global cell indices (c0 + c)
    autoPtr<Function1<scalar>> Tf;
};

struct SolidProps
{
    scalar rho = 0, Cp = 0, kappa = 0, Tref = 0, Hsref = 0;
};

struct SpecieProps
{
    word name;
    scalar Cp = 0, Tref = 0, Hsref = 0, mu = 0, Pr = 1;
    scalar rho0 = 0, T0 = 0, beta = 0;
};

// heRhoThermo, pureMixture or multiComponentMixture, const transport, hConst, Boussinesq or rhoConst,
// sensibleEnthalpy (psi = 0); hConst: hs = Cp (T - Tref) + Href; kappa = Cp mu / Pr
struct FluidProps
{
    scalar Cp = 0, Tref = 0, Hsref = 0, mu = 0, Pr = 1, kappa = 0;
    scalar rho0 = 0, T0 = 0, beta = 0;        // rho = rho0 (1 - beta (T - T0)); rhoConst: beta = 0
    bool dpdt = true;
    bool multiComponent = false;
    label inertIndex = -1;
    List<SpecieProps> species;
    bool turbulent = false;
    gc::TurbIn turb;
    scalar rho(scalar T) const { return rho0*(1 - beta*(T - T0)); }
};

class EnergyCoupling
{
public:
    EnergyCoupling(Time& runTime, List<Region>& regions, const gc::MeshIn& mi, gc::Engine& E);

    // time-dependent boundary values, fvOptions heat sources, and fixedTemperatureConstraint for the next energy solve
    void updateBCs(bool initCall = false);
    // fvOptions scalarSemiImplicitSource on species of region r
    void updateSpecieSources(label r, int fluidId) const;
    // per-region equation relaxation (solids always use "h": hEqn.relax() runs before the final-iteration flag
    // is set in solveSolid.H; fluids use h / hFinal)
    void setRelax(bool finalIter);
    gc::SolverCtl solverCtl(bool finalIter) const;
    word fieldName() const { return "h"; }
    void logMinMax() const;
    scalar solidDiffNo(scalar dt, scalar& meanDi) const;
    // device -> OpenFOAM fields (internal values and boundary state) before runTime.write()
    void writeBack();

    bool hasBuilding() const { return hasBuilding_; }
    const FluidProps& fluidProps(label r) const { return fluid_[r]; }
    const volScalarField& T(label r) const { return T_[r]; }
    volScalarField& T(label r) { return T_[r]; }

private:
    Time& runTime_;
    List<Region>& R_;
    const gc::MeshIn& mi_;
    gc::Engine& E_;
    PtrList<volScalarField> T_;
    PtrList<volScalarField> pSolid_;
    List<SolidProps> solid_;
    List<FluidProps> fluid_;
    List<TPatch> patches_;
    PtrList<SemiImplicitSource> hSources_;
    PtrList<SemiImplicitSource> ySources_;
    PtrList<FixedTempConstraint> tConstraints_;
    bool hasBuilding_ = false;
    label nZones_ = 0;
    PtrList<RadSource> radSources_;
    label nIr_ = 0;

    void readSolidProps(label r);
    void readFluidProps(label r);
    void readFvOptions();
    void setupPatches();
    void setupBuilding();
    void setupSchemes();
};

// the boundary condition as OpenFOAM interpreted it (regex / group entries resolved), as a dictionary
dictionary patchDict(const fvPatchScalarField& pf);
dictionary patchDict(const fvPatchVectorField& pf);
// "Gauss linear corrected | uncorrected | limited [corrected] psi | orthogonal" -> psi
scalar laplacianPsi(const fvMesh& mesh, const word& name);
// snGrad scheme "corrected | uncorrected | limited [corrected] psi" -> psi
scalar snGradPsi(const fvMesh& mesh, const word& name);
// "leastSquares" (0) / "cellLimited leastSquares k" (k); other schemes stop with an error
scalar gradLimitK(const fvMesh& mesh, const word& name);
// convection scheme "[bounded] Gauss upwind | linearUpwind grad(<field>) | linear | limitedLinear k":
// returns 0 upwind, 1 linearUpwind, 2 linear, 3 limitedLinear (k in coef); allowed: bit mask of accepted kinds
int divScheme(const fvMesh& mesh, const word& name, const word& field, int allowed, bool& bounded, scalar& coef);
// fvSolution solver entry -> device solver settings (Py version mapping)
gc::SolverCtl readSolver(const fvMesh& mesh, const word& name, bool pressure, bool positive = false);

} // namespace gpuCht
} // namespace Foam

#endif
