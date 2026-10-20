// Interface between the OpenFOAM application (host, OpenFOAM types) and the compute library (device).
// Only standard C++ types here: the application is compiled by wmake/g++, the library by nvcc or g++ -fopenmp.
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace gc
{

// ---------------------------------------------------------------------------- global multi-region mesh
// Cells: region by region. Internal faces: the faces inside each region (region by region, OpenFOAM order), then
// the interface faces of each coupled patch pair (side A -> side B orientation). Boundary faces: the non-coupled,
// non-empty patches region by region (empty patches carry no faces in OpenFOAM fields and are left out).
// All geometry is OpenFOAM's own (surfaceInterpolation, leastSquaresVectors, fvPatch deltaCoeffs).
struct MeshIn
{
    int nC = 0, nI = 0, nB = 0, nIface = 0;   // interface faces are internal faces [nI - nIface, nI)

    std::vector<int> own;                     // nI + nB
    std::vector<int> nei;                     // nI
    std::vector<double> V, C;                 // nC, 3 nC
    std::vector<double> Sf, Cf;               // 3 (nI + nB)
    std::vector<double> w;                    // nI: owner interpolation weight (region faces)
    std::vector<double> nonOrthDC;            // nI: nonOrthDeltaCoeffs (region faces)
    std::vector<double> corrVec;              // 3 nI: nonOrthCorrectionVectors (region faces)
    std::vector<double> lsP, lsN;             // 3 nI: leastSquares pVectors / nVectors (region faces);
                                              //       interface faces: boundary vectors of the side A / B patch face
    std::vector<double> dcI_A, dcI_B;         // nIface: patch deltaCoeffs of the side A / side B patch face
    std::vector<double> dcB;                  // nB: patch deltaCoeffs
    std::vector<double> lsB;                  // 3 nB: boundary leastSquares vectors

    struct Region
    {
        std::string name;
        bool fluid = false;
        int c0 = 0, nC = 0;                   // cells [c0, c0 + nC)
        int f0 = 0, nI = 0;                   // region internal faces [f0, f0 + nI)
        int b0 = 0, nB = 0;                   // region boundary faces [b0, b0 + nB) (global boundary numbering)
    };
    std::vector<Region> regions;

    struct Patch
    {
        std::string region, name, type;
        int b0 = 0, n = 0;                    // global boundary faces [b0, b0 + n)
    };
    std::vector<Patch> patches;

    struct Pair
    {
        int rA = 0, rB = 0;
        std::string pA, pB;
        int f0 = 0, n = 0;                    // global internal faces [f0, f0 + n)
    };
    std::vector<Pair> pairs;
};

// ---------------------------------------------------------------------------- linear solver settings
struct SolverCtl
{
    enum Method { PCG, PBICGSTAB, SMOOTH } method = PBICGSTAB;
    enum Precond { NONE, JACOBI, AMG, DILU } precond = JACOBI;
    // SMOOTH (smoothSolver): GS Gauss-Seidel, SYMGS forward + backward sweep, DILUSM x += DILU^-1 (b - A x);
    // nSweeps sweeps between residual checks
    enum Smoother { GS, SYMGS, DILUSM } smoother = GS;
    int nSweeps = 1;
    double tol = 1e-6, relTol = 0.0;
    int maxIter = 1000, minIter = 0;
    std::string name;                         // for the log (OpenFOAM-style solver name)
};

struct SolveInfo
{
    double res0 = 0, res = 0;
    int iters = 0;
};

// ---------------------------------------------------------------------------- energy boundary kinds
// T on boundary faces: T_b = a + b T_P with
//   FIXED: a = val, b = 0; ZEROGRAD: a = 0, b = 1; FIXEDGRAD: a = grad/dc, b = 1;
//   MIXED (OpenFOAM mixed): a = vf val + (1 - vf) grad/dc, b = 1 - vf;
//   INLETOUTLET: vf = (phi < 0 ? 1 : 0), a = vf val, b = 1 - vf
enum BKind
{
    B_FIXED = 0, B_ZEROGRAD = 1, B_FIXEDGRAD = 2, B_MIXED = 3, B_INLETOUTLET = 4,
    B_TURB_INTENSITY = 5, B_TURB_MIXINGLEN = 6
};
enum NutKindIn { NUT_CALC = 0, NUT_ZEROGRAD = 1, NUT_WALL = 2 };
enum AlphatKindIn { AT_CALC = 0, AT_ZEROGRAD = 1, AT_WALL = 2, AT_JAYATILLEKE = 3 };

// arrays exchanged with the engine
enum class A
{
    // per cell
    T, rho, cp, e, kappa, alphat, Su, Sp,
    // per region: cellLimited coefficient of grad(he) (<= 0 none), laplacian limited-corrected psi
    // (1 corrected, 0 uncorrected), energy equation relaxation factor (<= 0 no relaxation)
    rLimitK, rPsi, rRelax,
    // per boundary face
    bVal, bGrad, bVf, cpB, eB, kappaB, alphatB, TbB,
    // per interface face
    Rlayer, qIface, kappaWall, TA, TB,
    // per face (nI + nB): fluid mass flux for the energy convection
    phi
};

// ---------------------------------------------------------------------------- fluid regions
// Slots: the region's boundary faces (global boundary order), then its interface faces (walls for the flow)
struct SpecieIn
{
    std::string name;
    double Cp = 0, Hsref = 0, Tref = 298.15, mu = 0, Pr = 1;
    double rho0 = 0, T0 = 0, beta = 0;
    double limitK = 0;                        // grad(Yi) cellLimited coefficient
    double psiY = 1;                          // laplacian(muEff,Yi) limited-corrected psi
};

struct TurbIn
{
    bool enabled = false;
    double Cmu = 0.0845, C1 = 1.42, C2 = 1.68, C3 = 0.0;
    double sigmak = 0.71942, sigmaEps = 0.71942, eta0 = 4.38, beta = 0.012;
    double Prt = 1.0;                         // EddyDiffusivity::correctNut cell Prt
    double kMin = 1e-15, epsMin = 1e-15;
    int convK = 1, convEps = 1;               // 0 upwind, 1 linearUpwind, 2 linear, 3 limitedLinear
    double convKK = 1.0, convEpsK = 1.0;
    bool boundedK = false, boundedEps = false;
    double limitKK = 0.0, limitKEps = 0.0;
    double psiK = 1.0, psiEps = 1.0;
};

struct FluidIn
{
    int region = -1;                          // index in MeshIn.regions
    std::vector<int> ifFaces;                 // interface faces of the region (global internal face ids)
    std::vector<char> ifSideA;                // the fluid cell is the owner of that face
    double g[3] = {0, 0, 0};
    double ghRef = 0;
    bool solveDir[3] = {true, true, true};    // false: direction normal to empty patches (2-D)
    int conv = 1;                             // div(phi,U): 0 upwind, 1 linearUpwind grad(U)
    bool bounded = false;
    double limitKU = 0;                       // grad(U) cellLimited coefficient
    double psiU = 1, psiP = 1, psiSnP = 1, psiSnRho = 1;
    int pRefCell = -1;                        // local cell; -1 when p_rgh has a fixed-value patch
    double pRefValue = 0;
    double Cp = 0, Hsref = 0, Tref = 298.15, kappa = 0;   // pureMixture thermo for slot cpB/eB/kappaB
    // multiComponentMixture (empty: pureMixture)
    std::vector<SpecieIn> species;
    int inertIndex = -1;
    int convY = 0;                            // div(phi,Yi_h): 0 upwind, 3 limitedLinear
    double convYK = 1.0;                      // limitedLinear coefficient
    bool boundedY = false;
    double limitKH = 0;                       // grad(h) cellLimited coefficient of this region
    TurbIn turb;
};

// fluid arrays: cells (3 per cell for vectors), slots, internal faces of the region
enum class F
{
    U, p, pAbs, rho, dpdt, mu, eRho0, eT0, eBeta,               // cells
    k, eps, nut, alphat,                                        // turbulence cell fields
    Ub, rhoB, pAbsB, phiB, muB, eRho0B, eT0B, eBetaB,           // slots
    kb, epsb, nutB, alphatB,                                    // turbulence slot fields
    uVal, uVf, pVal, p0Val, pGrad,                              // slot boundary data
    phi                                                         // region internal faces
};
enum UKindIn { UK_FIXED = 0, UK_ZEROGRAD = 1, UK_INLETOUTLET = 2, UK_PIOV = 3, UK_SLIP = 4 };
enum PKindIn { PK_FIXED = 0, PK_ZEROGRAD = 1, PK_FLUX = 2, PK_PRGH_P = 3, PK_PRGH_TOTALP = 4, PK_TOTALP = 5 };

struct FlowPatchIn
{
    int s0 = 0, n = 0;                        // contiguous slot range [s0, s0 + n)
    bool mass = true;                         // true: massFlowRate, false: volumetricFlowRate
    bool outlet = false;                      // false: flowRateInletVelocity, true: flowRateOutletVelocity
    double flowRate = 0.0;
};

// ---------------------------------------------------------------------------- building physics (M6)
struct WeatherState
{
    double t = 0, Ta = 300, Tsky = 290, p = 101325, w = 0.01, wind = 0;
    double DNI = 0, DHI = 0, GHI = 0, rhoGround = 0.2;
    double sun[3] = {0, 0, 0};
};

struct BuildingIn
{
    std::string weatherFile;
    double groundReflectance = 0.2;
    bool shading = true;
    double pixel = 0.05;
    double solarUpdateInterval = 300.0;
    double transmittance = 0.81;
    double glassAbsorptance = 0.12;
    double interiorAbsorptance = 0.7;
    bool interiorRad = true;
    double emissivity = 0.9;
    double glassEmissivity = 0.84;
    int nZones = 0;
    // exterior faces (subset of global boundary faces [0, nB))
    std::vector<int> eb;
    std::vector<double> eHc, eEps, eAlpha;
    std::vector<char> eGlass;
    // fluid-solid interface faces (subset of global interface faces [0, nIface))
    std::vector<int> iPos;
    std::vector<double> iSign;                // +1 if solid is side A (Sf points solid -> air), -1 if air is side A
    std::vector<int> iZone;
    std::vector<char> iGlass;
    std::vector<char> iAirOwner;
};

class Engine
{
public:
    explicit Engine(const MeshIn& m);
    ~Engine();

    int nC() const;
    int nI() const;
    int nB() const;

    // array exchange (offset/count in the array's own index space)
    void put(A a, const std::vector<double>& v, long offset = 0);
    std::vector<double> get(A a, long offset = 0, long n = -1) const;
    void minMax(A a, long offset, long n, double& mn, double& mx) const;
    void putFluid(const std::vector<char>& fluid);        // per cell
    void putBKind(const std::vector<int>& kind, long offset = 0);

    // ---- energy
    void energyInit();                                    // after all per-cell / boundary data are in
    void energyStoreOld();                                // T, rho, he of the old time level
    SolveInfo energySolve(double dt, const SolverCtl& ctl, bool relaxOn);   // dt <= 0: steady
    void setFixedT(const std::vector<int>& cells, const std::vector<double>& T);   // fixedTemperatureConstraint
    std::vector<double> boundaryT() const;                // T on boundary faces
    // interface faces: T on the side A / side B patch faces, side conductances (W/K) A / B
    void interfaceState(std::vector<double>& TA, std::vector<double>& TB,
                        std::vector<double>& gA, std::vector<double>& gB) const;

    void energyBoundaryState();               // refresh boundary / interface T after a change of T
    // per region min / max of T over its cells and patch faces (boundary faces, its side of the interfaces), as
    // OpenFOAM's min(T) / max(T) of the GeometricField; the boundary state as the last solve left it
    void energyMinMaxT(std::vector<double>& mn, std::vector<double>& mx) const;
    void setEnergyConvection(int conv, double k, bool bounded);   // 0 upwind 1 linearUpwind 2 linear 3 limitedLinear

    // ---- fvOptions scalarSemiImplicitSource evaluated on the device. Source k adds su_k w_ki to Su and sp_k w_ki
    // to Sp of its cells c_ki (summed over the sources in order k, as the host loop did). sourceSet, once: the cell
    // lists (global cells for the energy, fluidId < 0; cells of fluid fluidId for its specie sp) and weights; the
    // other cells keep Su = Sp = 0. sourceValues, at every update: su_k, sp_k of all sources of the set.
    int sourceSet(int fluidId, int sp, const std::vector<std::vector<int>>& cells,
                  const std::vector<std::vector<double>>& w);
    void sourceValues(int set, const std::vector<double>& su, const std::vector<double>& sp);

    // ---- building physics
    void buildingInit(const BuildingIn& bi, int& nGlaze, int& nGlazeMiss, double& areaExt, double& areaGlaze);
    void buildingUpdate(double t, const std::vector<double>& zoneQrad);
    WeatherState buildingWeather(double t) const;

    // ---- fluids
    int addFluid(const FluidIn& fi);
    int nSlots(int id) const;
    void fput(int id, F a, const std::vector<double>& v, long offset = 0);
    std::vector<double> fget(int id, F a, long offset = 0, long n = -1) const;
    void fminMax(int id, F a, double& mn, double& mx) const;   // over the whole array
    // cell array a at the cells of slots [0, n) (a[sCell[s]])
    std::vector<double> fgetAtSlotCells(int id, F a, long n) const;
    void fputKinds(int id, const std::vector<int>& uKind, const std::vector<int>& pKind,
                   const std::vector<char>& uAssign, const std::vector<char>& uFixes);
    void fputFlowPatches(int id, const std::vector<FlowPatchIn>& fp);
    void fputFlowRate(int id, int fpIndex, double flowRate);
    void fputSpecie(int id, int sp, const std::vector<double>& Y, const std::vector<int>& kind,
                    const std::vector<double>& val, const std::vector<double>& grad, const std::vector<double>& vf);
    void fputSpecieVal(int id, int sp, const std::vector<double>& val, long offset = 0);
    void fputSpecieSource(int id, int sp, const std::vector<double>& Su, const std::vector<double>& Sp);
    void fgetSpecie(int id, int sp, std::vector<double>& Y, std::vector<double>& Yb) const;
    std::vector<double> fgetSpecieSlots(int id, int sp, long offset, long n) const;
    void fluidUpdateMixture(int id);
    void fputTurb(int id, const std::vector<double>& k, const std::vector<double>& eps,
                  const std::vector<double>& nut, const std::vector<double>& alphat,
                  const std::vector<int>& kKind, const std::vector<double>& kVal,
                  const std::vector<int>& epsKind, const std::vector<double>& epsVal,
                  const std::vector<int>& nutKind, const std::vector<int>& atKind,
                  const std::vector<double>& eCmu, const std::vector<double>& eKappa,
                  const std::vector<double>& nCmu, const std::vector<double>& nKappa, const std::vector<double>& nE,
                  const std::vector<double>& atPrt, const std::vector<double>& atKappa, const std::vector<double>& atE,
                  const std::vector<double>& sY, const std::vector<double>& alphatB0);
    // initial state as createFluidFields: rho = thermo rho(T); phi = interpolate(rho U).Sf unless phiGiven;
    // turbulence->validate() when turbulent
    void fluidStart(int id, bool phiGiven);
    void fluidStoreOld(int id, double dt, bool frozen, bool storePrevIter);
    void fluidMomentum(int id, double dt, double alphaU, const SolverCtl& ctl, bool predictor, SolveInfo info[3]);
    void fluidSpecies(int id, double dt, const std::vector<double>& alphaY, const SolverCtl& ctl,
                      std::vector<SolveInfo>& info);
    // sameRho: the temperature has not changed since the previous call (a further corrector of the same outer
    // iteration), so rho and grad(rho) are those of that call
    SolveInfo fluidPressure(int id, double dt, const SolverCtl& ctl, double pRelax, int nNonOrth, double contErr[2],
                            bool sameRho = false);
    void fluidTurbulence(int id, double dt, double alphaK, double alphaEps, const SolverCtl& ctlK,
                         const SolverCtl& ctlEps, SolveInfo& infoEps, SolveInfo& infoK);
    void fluidUpdateRho(int id);
    void fluidUpdateBoundary(int id, bool evalFlowPatches = true);
    void fluidToEnergy(int id, double dt, bool dpdt);
    // the fluid's current rho -> energy rho of its cells (call before energyStoreOld: rho.oldTime() of the
    // fluid EEqn is the density at the end of the previous time step, before rhoEqn)
    void fluidRhoToEnergy(int id);
    double fluidCourantNo(int id, double dt, double& meanCo) const;

    double memGB() const;
    void sync() const;                        // wait for the device (phase timing)
    std::string profReport() const;           // GPUCHT_PROF=1: wall time per library scope (else empty)

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace gc
