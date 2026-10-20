#include "energyCoupling.H"

#include "addToRunTimeSelectionTable.H"
#include "emptyFvPatch.H"
#include "fixedGradientFvPatchFields.H"
#include "mixedFvPatchFields.H"
#include "physicoChemicalConstants.H"
#include "Tuple2.H"
#include "IStringStream.H"
#include "OStringStream.H"

#include <set>

namespace Foam
{

makePatchTypeField(fvPatchScalarField, exteriorSurfaceFvPatchScalarField);

exteriorSurfaceFvPatchScalarField::exteriorSurfaceFvPatchScalarField
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF
)
:
    mixedFvPatchScalarField(p, iF)
{
    refValue() = Zero;
    refGrad() = Zero;
    valueFraction() = 0.0;
}

exteriorSurfaceFvPatchScalarField::exteriorSurfaceFvPatchScalarField
(
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const dictionary& dict
)
:
    mixedFvPatchScalarField(p, iF),
    hc_(dict.get<scalar>("hc"))
{
    hasEps_ = dict.readIfPresent("emissivity", eps_);
    hasAlpha_ = dict.readIfPresent("absorptance", alpha_);
    if (dict.found("value"))
    {
        fvPatchScalarField::operator=(scalarField("value", dict, p.size()));
    }
    else
    {
        fvPatchScalarField::operator=(300.0);
    }
    refValue() = *this;
    refGrad() = Zero;
    valueFraction() = 0.0;
}

exteriorSurfaceFvPatchScalarField::exteriorSurfaceFvPatchScalarField
(
    const exteriorSurfaceFvPatchScalarField& ptf,
    const fvPatch& p,
    const DimensionedField<scalar, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    mixedFvPatchScalarField(ptf, p, iF, mapper),
    hc_(ptf.hc_),
    hasEps_(ptf.hasEps_),
    hasAlpha_(ptf.hasAlpha_),
    eps_(ptf.eps_),
    alpha_(ptf.alpha_)
{}

exteriorSurfaceFvPatchScalarField::exteriorSurfaceFvPatchScalarField
(
    const exteriorSurfaceFvPatchScalarField& ptf
)
:
    mixedFvPatchScalarField(ptf),
    hc_(ptf.hc_),
    hasEps_(ptf.hasEps_),
    hasAlpha_(ptf.hasAlpha_),
    eps_(ptf.eps_),
    alpha_(ptf.alpha_)
{}

exteriorSurfaceFvPatchScalarField::exteriorSurfaceFvPatchScalarField
(
    const exteriorSurfaceFvPatchScalarField& ptf,
    const DimensionedField<scalar, volMesh>& iF
)
:
    mixedFvPatchScalarField(ptf, iF),
    hc_(ptf.hc_),
    hasEps_(ptf.hasEps_),
    hasAlpha_(ptf.hasAlpha_),
    eps_(ptf.eps_),
    alpha_(ptf.alpha_)
{}

void exteriorSurfaceFvPatchScalarField::write(Ostream& os) const
{
    fvPatchScalarField::write(os);
    os.writeEntry("hc", hc_);
    if (hasEps_) os.writeEntry("emissivity", eps_);
    if (hasAlpha_) os.writeEntry("absorptance", alpha_);
    writeEntry("value", os);
}

namespace gpuCht
{

static const scalar Tstd = 298.15;

// the boundary condition as OpenFOAM interpreted it (regex / group entries resolved), as a dictionary
dictionary patchDict(const fvPatchScalarField& pf)
{
    OStringStream os;
    pf.write(os);
    IStringStream is(os.str());
    return dictionary(is);
}

dictionary patchDict(const fvPatchVectorField& pf)
{
    OStringStream os;
    pf.write(os);
    IStringStream is(os.str());
    return dictionary(is);
}

static bool isCoupledT(const word& t)
{
    return t == "compressible::turbulentTemperatureCoupledBaffleMixed"
        || t == "turbulentTemperatureCoupledBaffleMixed"
        || t == "compressible::turbulentTemperatureRadCoupledMixed"
        || t == "turbulentTemperatureRadCoupledMixed";
}

// ----------------------------------------------------------------------------------------- settings parsing
// "corrected | uncorrected | limited [corrected] psi" from the stream -> psi
static scalar readSnGradPsi(ITstream& is, const word& where)
{
    const word sn(is);
    if (sn == "corrected") return 1;
    if (sn == "uncorrected") return 0;
    if (sn == "limited")
    {
        token t(is);
        if (t.isWord())
        {
            if (t.wordToken() != "corrected")
                FatalErrorInFunction << where << ": limited " << t.wordToken() << " is not supported"
                    << exit(FatalError);
            t = token(is);
        }
        if (!t.isNumber())
            FatalErrorInFunction << where << ": limited needs a coefficient" << exit(FatalError);
        return t.number();
    }
    FatalErrorInFunction << where << ": snGrad " << sn << " is not supported (corrected, uncorrected, limited)"
        << exit(FatalError);
    return 0;
}

scalar laplacianPsi(const fvMesh& mesh, const word& name)
{
    ITstream& is = mesh.laplacianScheme(name);
    const word g(is);
    if (g != "Gauss")
        FatalErrorInFunction << mesh.name() << ": laplacian scheme for " << name << " must be Gauss" << exit(FatalError);
    const word interp(is);
    if (interp != "linear")
        FatalErrorInFunction << mesh.name() << ": laplacian interpolation for " << name << " must be linear, got "
            << interp << exit(FatalError);
    return readSnGradPsi(is, mesh.name() + ": laplacian " + name);
}

scalar snGradPsi(const fvMesh& mesh, const word& name)
{
    ITstream& is = mesh.snGradScheme(name);
    return readSnGradPsi(is, mesh.name() + ": " + name);
}

int divScheme(const fvMesh& mesh, const word& name, const word& field, int allowed, bool& bounded, scalar& coef)
{
    ITstream& is = mesh.divScheme(name);
    const word where = mesh.name() + ": " + name;
    word w(is);
    bounded = false;
    coef = 1;
    if (w == "bounded")
    {
        bounded = true;
        w = word(is);
    }
    if (w != "Gauss")
        FatalErrorInFunction << where << ": must be [bounded] Gauss <scheme>" << exit(FatalError);
    const word s(is);
    int k = -1;
    if (s == "upwind") k = 0;
    else if (s == "linearUpwind")
    {
        const word gname(is);
        if (gname != "grad(" + field + ")")
            FatalErrorInFunction << where << ": linearUpwind must use grad(" << field << "), got " << gname
                << exit(FatalError);
        k = 1;
    }
    else if (s == "linear") k = 2;
    else if (s == "limitedLinear")
    {
        token t(is);
        if (!t.isNumber())
            FatalErrorInFunction << where << ": limitedLinear needs a coefficient" << exit(FatalError);
        coef = t.number();
        k = 3;
    }
    if (k < 0 || !(allowed & (1 << k)))
        FatalErrorInFunction << where << ": scheme " << s << " is not supported here" << exit(FatalError);
    return k;
}

scalar gradLimitK(const fvMesh& mesh, const word& name)
{
    ITstream& is = mesh.gradScheme(name);
    const word s(is);
    if (s == "leastSquares") return 0;
    if (s == "cellLimited")
    {
        const word b(is);
        if (b != "leastSquares")
            FatalErrorInFunction << mesh.name() << ": " << name << ": cellLimited " << b
                << " is not supported (cellLimited leastSquares k)" << exit(FatalError);
        token t(is);
        return t.number();
    }
    FatalErrorInFunction << mesh.name() << ": gradient scheme " << s << " for " << name
        << " is not supported (leastSquares, cellLimited leastSquares k)" << exit(FatalError);
    return 0;
}

gc::SolverCtl readSolver(const fvMesh& mesh, const word& name, bool pressure, bool positive)
{
    const dictionary& d = mesh.solverDict(name);
    const word s = d.get<word>("solver");
    gc::SolverCtl c;
    c.tol = d.getOrDefault<scalar>("tolerance", 1e-6);
    c.relTol = d.getOrDefault<scalar>("relTol", 0);
    c.maxIter = d.getOrDefault<label>("maxIter", 1000);
    c.minIter = d.getOrDefault<label>("minIter", 0);
    if (pressure || s == "GAMG")
    {
        c.method = pressure ? gc::SolverCtl::PCG : gc::SolverCtl::PBICGSTAB;
        c.precond = gc::SolverCtl::AMG;
        c.name = pressure ? "AMGPCG" : "AMGBiCGStab";
    }
    else if (s == "PCG" || s == "PPCG" || s == "PPCR")
    {
        c.method = gc::SolverCtl::PCG;
        c.precond = d.getOrDefault<word>("preconditioner", "DIC") == "GAMG" ? gc::SolverCtl::AMG : gc::SolverCtl::JACOBI;
        c.name = s;
    }
    else if (s == "PBiCG" || s == "PBiCGStab")
    {
        c.method = gc::SolverCtl::PBICGSTAB;
        c.precond = gc::SolverCtl::JACOBI;
        c.name = "PBiCGStab";
    }
    else if (s == "smoothSolver")
    {
        c.method = positive ? gc::SolverCtl::SMOOTH : gc::SolverCtl::PBICGSTAB;
        c.precond = gc::SolverCtl::JACOBI;
        c.name = positive ? "smoothSolver" : "PBiCGStab";
    }
    else
    {
        FatalErrorInFunction << mesh.name() << ": solver " << s << " for " << name
            << " is not supported (PCG, PBiCG(Stab), smoothSolver, GAMG)" << exit(FatalError);
    }
    return c;
}

// ----------------------------------------------------------------------------------------- setup
EnergyCoupling::EnergyCoupling(Time& runTime, List<Region>& regions, const gc::MeshIn& mi, gc::Engine& E)
:
    runTime_(runTime),
    R_(regions),
    mi_(mi),
    E_(E),
    T_(regions.size()),
    pSolid_(regions.size()),
    solid_(regions.size()),
    fluid_(regions.size())
{
    nIr_ = mi.nI - mi.nIface;
    forAll(R_, r)
    {
        fvMesh& mesh = *R_[r].mesh;
        T_.set(r, new volScalarField(IOobject("T", runTime.timeName(), mesh, IOobject::MUST_READ,
                                              IOobject::AUTO_WRITE), mesh));
        if (R_[r].fluid)
        {
            readFluidProps(r);
            continue;
        }
        IOobject pio("p", runTime.timeName(), mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE);
        if (pio.typeHeaderOk<volScalarField>(true))
            pSolid_.set(r, new volScalarField(pio, mesh));
        readSolidProps(r);
    }
    // per-cell properties (he = Cp (T - Tref) + Href)
    const label nC = mi.nC;
    std::vector<double> T(nC), rho(nC), cp(nC), e(nC), kappa(nC);
    std::vector<char> fluid(nC, 0);
    forAll(R_, r)
    {
        const label c0 = R_[r].c0;
        const scalarField& Ti = T_[r].primitiveField();
        const bool fl = R_[r].fluid;
        const SolidProps& s = solid_[r];
        const FluidProps& f = fluid_[r];
        forAll(Ti, c)
        {
            T[c0 + c] = Ti[c];
            if (fl)
            {
                rho[c0 + c] = f.rho(Ti[c]);
                cp[c0 + c] = f.Cp;
                e[c0 + c] = f.Hsref - f.Cp*f.Tref;
                kappa[c0 + c] = f.kappa;
                fluid[c0 + c] = 1;
            }
            else
            {
                rho[c0 + c] = s.rho;
                cp[c0 + c] = s.Cp;
                e[c0 + c] = s.Hsref - s.Cp*s.Tref;
                kappa[c0 + c] = s.kappa;
            }
        }
    }
    E_.put(gc::A::T, T);
    E_.put(gc::A::rho, rho);
    E_.put(gc::A::cp, cp);
    E_.put(gc::A::e, e);
    E_.put(gc::A::kappa, kappa);
    E_.putFluid(fluid);
    // boundary faces carry the properties of their region (pure substances)
    const label nB = mi.nB;
    std::vector<double> cpB(nB), eB(nB), kB(nB);
    forAll(R_, r)
    {
        const bool fl = R_[r].fluid;
        const scalar Cp = fl ? fluid_[r].Cp : solid_[r].Cp;
        const scalar ee = fl ? fluid_[r].Hsref - fluid_[r].Cp*fluid_[r].Tref : solid_[r].Hsref - solid_[r].Cp*solid_[r].Tref;
        const scalar k = fl ? fluid_[r].kappa : solid_[r].kappa;
        for (label b = R_[r].b0; b < R_[r].b0 + mi.regions[r].nB; ++b)
        {
            cpB[b] = Cp;
            eB[b] = ee;
            kB[b] = k;
        }
    }
    E_.put(gc::A::cpB, cpB);
    E_.put(gc::A::eB, eB);
    E_.put(gc::A::kappaB, kB);
    readFvOptions();
    setupPatches();
    setupSchemes();
    E_.energyInit();
    // initial coupled interface T values from 0/<region>/T (used by fluid thermo before the first energy solve)
    if (mi.nIface > 0)
    {
        std::vector<double> TA = E_.get(gc::A::TA), TB = E_.get(gc::A::TB);
        for (const TPatch& tp : patches_)
        {
            if (tp.kind != TPatch::COUPLED || !tp.n) continue;
            const PatchMap& pm = R_[tp.region].patches[tp.patch];
            const scalarField& pf = T_[tp.region].boundaryField()[tp.patch];
            std::vector<double>& dst = pm.kind == PatchMap::IFACE_A ? TA : TB;
            forAll(pf, i) dst[pm.ifaceIdx[i]] = pf[i];
        }
        E_.put(gc::A::TA, TA);
        E_.put(gc::A::TB, TB);
    }
}

static SpecieProps readSpecieDict(const word& name, const dictionary& d, const word& eos, const word& where)
{
    SpecieProps s;
    s.name = name;
    const dictionary& tr = d.subDict("transport");
    s.mu = tr.get<scalar>("mu");
    s.Pr = tr.get<scalar>("Pr");
    const dictionary& th = d.subDict("thermodynamics");
    s.Cp = th.get<scalar>("Cp");
    s.Tref = th.getOrDefault<scalar>("Tref", Tstd);
    s.Hsref = th.getOrDefault<scalar>("Href", 0);
    if (th.getOrDefault<scalar>("Hf", 0) != 0)
        FatalErrorInFunction << where << "/" << name << ": non-zero Hf is not used by sensibleEnthalpy (use Href)"
            << exit(FatalError);
    const dictionary& es = d.subDict("equationOfState");
    if (eos == "Boussinesq")
    {
        s.rho0 = es.get<scalar>("rho0");
        s.T0 = es.get<scalar>("T0");
        s.beta = es.get<scalar>("beta");
    }
    else if (eos == "rhoConst")
    {
        s.rho0 = es.get<scalar>("rho");
        s.T0 = 0;
        s.beta = 0;
    }
    else
    {
        FatalErrorInFunction << where << ": equationOfState " << eos << " is not supported (Boussinesq, rhoConst)"
            << exit(FatalError);
    }
    return s;
}

void EnergyCoupling::readFluidProps(label r)
{
    const fvMesh& mesh = *R_[r].mesh;
    const word where = mesh.name() + "/thermophysicalProperties";
    IOdictionary tp(IOobject("thermophysicalProperties", mesh.time().constant(), mesh, IOobject::MUST_READ,
                             IOobject::NO_WRITE, IOobject::NO_REGISTER));
    const dictionary& tt = tp.subDict("thermoType");
    const std::pair<const char*, const char*> req[] = {{"type", "heRhoThermo"}, {"transport", "const"},
        {"thermo", "hConst"}, {"energy", "sensibleEnthalpy"}};
    for (const auto& kv : req)
    {
        const word got = tt.getOrDefault<word>(kv.first, word::null);
        if (got != kv.second)
            FatalErrorInFunction << where << ": thermoType/" << kv.first << " " << got
                << " is not supported (need " << kv.second << ")" << exit(FatalError);
    }
    const word mixType = tt.get<word>("mixture");
    const word eos = tt.get<word>("equationOfState");
    FluidProps& f = fluid_[r];
    if (mixType == "pureMixture")
    {
        const SpecieProps s = readSpecieDict("mixture", tp.subDict("mixture"), eos, where);
        f.Cp = s.Cp;
        f.Tref = s.Tref;
        f.Hsref = s.Hsref;
        f.mu = s.mu;
        f.Pr = s.Pr;
        f.kappa = f.Cp*f.mu/f.Pr;
        f.rho0 = s.rho0;
        f.T0 = s.T0;
        f.beta = s.beta;
    }
    else if (mixType == "multiComponentMixture")
    {
        f.multiComponent = true;
        const wordList spNames(tp.get<wordList>("species"));
        if (spNames.empty())
            FatalErrorInFunction << where << ": empty species list" << exit(FatalError);
        const word inert = tp.getOrDefault<word>("inertSpecie", spNames[0]);
        f.species.setSize(spNames.size());
        f.inertIndex = -1;
        forAll(spNames, m)
        {
            f.species[m] = readSpecieDict(spNames[m], tp.subDict(spNames[m]), eos, where);
            if (spNames[m] == inert) f.inertIndex = m;
        }
        if (f.inertIndex < 0)
            FatalErrorInFunction << where << ": inertSpecie " << inert << " not in species " << spNames
                << exit(FatalError);
        const SpecieProps& s = f.species[f.inertIndex];
        f.Cp = s.Cp;
        f.Tref = f.species[0].Tref;
        f.Hsref = s.Hsref;
        f.mu = s.mu;
        f.Pr = s.Pr;
        f.kappa = f.Cp*f.mu/f.Pr;
        f.rho0 = s.rho0;
        f.T0 = s.T0;
        f.beta = s.beta;
    }
    else
    {
        FatalErrorInFunction << where << ": mixture " << mixType
            << " is not supported (pureMixture, multiComponentMixture)" << exit(FatalError);
    }
    f.dpdt = tp.getOrDefault<bool>("dpdt", true);
    IOdictionary turb(IOobject("turbulenceProperties", mesh.time().constant(), mesh, IOobject::READ_IF_PRESENT,
                               IOobject::NO_WRITE, IOobject::NO_REGISTER));
    const word st = turb.getOrDefault<word>("simulationType", "laminar");
    const word twhere = mesh.name() + "/turbulenceProperties";
    if (st == "laminar")
    {
        if (turb.isDict("laminar"))
        {
            const dictionary& lm = turb.subDict("laminar");
            const word mdl = lm.getOrDefault<word>("laminarModel", lm.getOrDefault<word>("model", "Stokes"));
            if (mdl != "Stokes")
                FatalErrorInFunction << twhere << ": laminar model " << mdl << " is not supported (Stokes)"
                    << exit(FatalError);
        }
    }
    else if (st == "RAS")
    {
        const dictionary& ras = turb.subDict("RAS");
        const word mdl = ras.getOrDefault<word>("RASModel", ras.getOrDefault<word>("model", word::null));
        if (ras.getOrDefault<bool>("turbulence", true))
        {
            if (mdl != "RNGkEpsilon")
                FatalErrorInFunction << twhere << ": RAS model " << mdl << " is not supported (RNGkEpsilon)"
                    << exit(FatalError);
            f.turbulent = true;
            f.turb.enabled = true;
            f.turb.kMin = ras.getOrDefault<scalar>("kMin", 1e-15);
            f.turb.epsMin = ras.getOrDefault<scalar>("epsilonMin", 1e-15);
            if (ras.isDict("RNGkEpsilonCoeffs"))
            {
                const dictionary& cd = ras.subDict("RNGkEpsilonCoeffs");
                f.turb.Cmu = cd.getOrDefault<scalar>("Cmu", f.turb.Cmu);
                f.turb.C1 = cd.getOrDefault<scalar>("C1", f.turb.C1);
                f.turb.C2 = cd.getOrDefault<scalar>("C2", f.turb.C2);
                f.turb.C3 = cd.getOrDefault<scalar>("C3", f.turb.C3);
                f.turb.sigmak = cd.getOrDefault<scalar>("sigmak", f.turb.sigmak);
                f.turb.sigmaEps = cd.getOrDefault<scalar>("sigmaEps", f.turb.sigmaEps);
                f.turb.eta0 = cd.getOrDefault<scalar>("eta0", f.turb.eta0);
                f.turb.beta = cd.getOrDefault<scalar>("beta", f.turb.beta);
                f.turb.Prt = cd.getOrDefault<scalar>("Prt", f.turb.Prt);
            }
        }
    }
    else
    {
        FatalErrorInFunction << twhere << ": simulationType " << st << " is not supported (laminar, RAS)"
            << exit(FatalError);
    }
    IOobject rio("radiationProperties", mesh.time().constant(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE,
                 IOobject::NO_REGISTER);
    if (rio.typeHeaderOk<IOdictionary>(true))
    {
        IOdictionary rp(rio);
        const word rm = rp.getOrDefault<word>("radiationModel", "none");
        if (rp.getOrDefault<bool>("radiation", rm != "none") && rm != "none")
            FatalErrorInFunction << mesh.name() << "/radiationProperties: radiationModel " << rm
                << " is not supported (none)" << exit(FatalError);
    }
    IOobject cio("combustionProperties", mesh.time().constant(), mesh, IOobject::MUST_READ, IOobject::NO_WRITE,
                 IOobject::NO_REGISTER);
    if (cio.typeHeaderOk<IOdictionary>(true))
    {
        IOdictionary cp(cio);
        const word cm = cp.getOrDefault<word>("combustionModel", "none");
        if (cp.getOrDefault<bool>("active", true) && cm != "none")
            FatalErrorInFunction << mesh.name() << "/combustionProperties: combustionModel " << cm
                << " is not supported (none)" << exit(FatalError);
    }
    Info<< "    " << mesh.name() << ": fluid " << mixType << " " << eos << " (" << f.species.size() << " species, "
        << (f.turbulent ? "RNGkEpsilon" : "laminar") << ")"
        << " rho0 " << f.rho0 << " T0 " << f.T0 << " beta " << f.beta << " Cp " << f.Cp << " mu " << f.mu
        << " Pr " << f.Pr << " dpdt " << f.dpdt << endl;
}

void EnergyCoupling::readSolidProps(label r)
{
    const fvMesh& mesh = *R_[r].mesh;
    IOdictionary tp(IOobject("thermophysicalProperties", mesh.time().constant(), mesh, IOobject::MUST_READ,
                             IOobject::NO_WRITE, IOobject::NO_REGISTER));
    const dictionary& tt = tp.subDict("thermoType");
    const std::pair<const char*, const char*> req[] = {{"type", "heSolidThermo"}, {"mixture", "pureMixture"},
        {"transport", "constIso"}, {"thermo", "hConst"}, {"equationOfState", "rhoConst"}, {"energy", "sensibleEnthalpy"}};
    for (const auto& kv : req)
    {
        const word got = tt.getOrDefault<word>(kv.first, word::null);
        if (got != kv.second)
            FatalErrorInFunction << mesh.name() << "/thermophysicalProperties: thermoType/" << kv.first << " " << got
                << " is not supported (need " << kv.second << ")" << exit(FatalError);
    }
    const dictionary& mx = tp.subDict("mixture");
    SolidProps& s = solid_[r];
    s.kappa = mx.subDict("transport").get<scalar>("kappa");
    const dictionary& th = mx.subDict("thermodynamics");
    s.Cp = th.get<scalar>("Cp");
    s.Tref = th.getOrDefault<scalar>("Tref", Tstd);
    s.Hsref = th.getOrDefault<scalar>("Href", 0);
    s.rho = mx.subDict("equationOfState").get<scalar>("rho");
    Info<< "    " << mesh.name() << ": solid rho " << s.rho << " Cp " << s.Cp << " kappa " << s.kappa << endl;
}

void EnergyCoupling::readFvOptions()
{
    DynamicList<SemiImplicitSource*> hList, yList;
    DynamicList<FixedTempConstraint*> tList;
    forAll(R_, r)
    {
        const fvMesh& mesh = *R_[r].mesh;
        const word docs[] = {"fvOptions", "fvModels", "fvConstraints"};
        const word dirs[] = {mesh.time().constant(), mesh.time().system()};
        for (const word& doc : docs)
        {
            for (const word& dir : dirs)
            {
                IOobject io(doc, dir, mesh, IOobject::MUST_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER);
                if (!io.typeHeaderOk<IOdictionary>(false) && !isFile(io.objectPath())) continue;
                IOdictionary d(io);
                const dictionary& opts = d.isDict("options") ? d.subDict("options") : d;
                const word where = dir + "/" + mesh.name() + "/" + doc;
                for (const entry& e : opts)
                {
                    if (!e.isDict()) continue;
                    const word& name = e.keyword();
                    if (name == "FoamFile") continue;
                    const dictionary& sd = e.dict();
                    if (!sd.getOrDefault<bool>("active", true)) continue;
                    const word t = sd.get<word>("type");
                    const word ewhere = where + "/" + name;
                    if (t == "fixedTemperatureConstraint")
                    {
                        const dictionary& c = sd.optionalSubDict("fixedTemperatureConstraintCoeffs");
                        const word mode = c.getOrDefault<word>("mode", "uniform");
                        if (mode != "uniform")
                            FatalErrorInFunction << ewhere << ": mode " << mode << " is not supported (uniform)"
                                << exit(FatalError);
                        const word sel = c.get<word>("selectionMode");
                        if (sel != "cellZone")
                            FatalErrorInFunction << ewhere << ": selectionMode " << sel
                                << " is not supported (cellZone)" << exit(FatalError);
                        const word zn = c.get<word>("cellZone");
                        const label zid = mesh.cellZones().findZoneID(zn);
                        if (zid < 0)
                            FatalErrorInFunction << ewhere << ": cellZone " << zn << " not found in " << mesh.name()
                                << exit(FatalError);
                        const labelList& zc = mesh.cellZones()[zid];
                        FixedTempConstraint* ft = new FixedTempConstraint();
                        ft->region = r;
                        ft->gcells.resize(zc.size());
                        forAll(zc, i) ft->gcells[i] = R_[r].c0 + zc[i];
                        ft->Tf = Function1<scalar>::New("temperature", c, &mesh);
                        tList.append(ft);
                        Info<< "    " << mesh.name() << ": fvOption " << name << " fixedTemperatureConstraint on "
                            << zn << " (" << zc.size() << " cells)" << endl;
                        continue;
                    }
                    if (t != "scalarSemiImplicitSource")
                        FatalErrorInFunction << ewhere << ": fvOption type " << t
                            << " is not supported (scalarSemiImplicitSource, fixedTemperatureConstraint)"
                            << exit(FatalError);
                    const dictionary& c = sd.optionalSubDict("scalarSemiImplicitSourceCoeffs");
                    const word sel = c.get<word>("selectionMode");
                    labelList cells;
                    if (sel == "all")
                    {
                        cells.setSize(mesh.nCells());
                        forAll(cells, i) cells[i] = i;
                    }
                    else if (sel == "cellZone")
                    {
                        const word zn = c.get<word>("cellZone");
                        const label zid = mesh.cellZones().findZoneID(zn);
                        if (zid < 0)
                            FatalErrorInFunction << ewhere << ": cellZone " << zn << " not found in " << mesh.name()
                                << exit(FatalError);
                        cells = mesh.cellZones()[zid];
                    }
                    else
                    {
                        FatalErrorInFunction << ewhere << ": selectionMode " << sel
                            << " is not supported (all, cellZone)" << exit(FatalError);
                    }
                    const word vmode = c.getOrDefault<word>("volumeMode", "absolute");
                    if (vmode != "absolute" && vmode != "specific")
                        FatalErrorInFunction << ewhere << ": volumeMode " << vmode
                            << " is not supported (absolute, specific)" << exit(FatalError);
                    const scalarField& V = mesh.V();
                    scalar Vsum = 0;
                    forAll(cells, i) Vsum += V[cells[i]];
                    std::vector<double> wV(cells.size());
                    forAll(cells, i) wV[i] = (vmode == "absolute") ? (Vsum > 0 ? V[cells[i]]/Vsum : 0.0) : V[cells[i]];

                    auto addField = [&](const word& fld, autoPtr<Function1<scalar>> SuF,
                                        autoPtr<Function1<scalar>> SpF, scalar SuC, scalar SpC)
                    {
                        SemiImplicitSource* src = new SemiImplicitSource();
                        src->region = r;
                        src->wV = wV;
                        src->SuF = std::move(SuF);
                        src->SpF = std::move(SpF);
                        src->SuConst = SuC;
                        src->SpConst = SpC;
                        if (fld == "h" || fld == "T")
                        {
                            src->specie = -1;
                            src->cells.resize(cells.size());
                            forAll(cells, i) src->cells[i] = R_[r].c0 + cells[i];
                            hList.append(src);
                            Info<< "    " << mesh.name() << ": fvOption " << name << " scalarSemiImplicitSource on "
                                << fld << " (" << cells.size() << " cells, " << vmode << ")" << endl;
                            return;
                        }
                        if (R_[r].fluid && fluid_[r].multiComponent)
                        {
                            forAll(fluid_[r].species, m)
                            {
                                if (fluid_[r].species[m].name == fld && m != fluid_[r].inertIndex)
                                {
                                    src->specie = m;
                                    src->cells.resize(cells.size());
                                    forAll(cells, i) src->cells[i] = cells[i];
                                    yList.append(src);
                                    Info<< "    " << mesh.name() << ": fvOption " << name
                                        << " scalarSemiImplicitSource on specie " << fld << " (" << cells.size()
                                        << " cells, " << vmode << ")" << endl;
                                    return;
                                }
                            }
                        }
                        delete src;
                        FatalErrorInFunction << ewhere << ": source on field " << fld
                            << " is not supported (h or an active specie)" << exit(FatalError);
                    };

                    if (c.isDict("sources"))
                    {
                        const dictionary& srcs = c.subDict("sources");
                        for (const entry& fe : srcs)
                        {
                            if (!fe.isDict()) continue;
                            const word& fld = fe.keyword();
                            const dictionary& fd = fe.dict();
                            autoPtr<Function1<scalar>> suF, spF;
                            if (fd.found("explicit")) suF = Function1<scalar>::New("explicit", fd, &mesh);
                            if (fd.found("implicit")) spF = Function1<scalar>::New("implicit", fd, &mesh);
                            addField(fld, std::move(suF), std::move(spF), 0, 0);
                        }
                    }
                    else if (c.isDict("injectionRateSuSp"))
                    {
                        const dictionary& inj = c.subDict("injectionRateSuSp");
                        for (const entry& fe : inj)
                        {
                            const word& fld = fe.keyword();
                            ITstream is(fe.stream());
                            const Tuple2<scalar, scalar> susp(is);
                            addField(fld, nullptr, nullptr, susp.first(), susp.second());
                        }
                    }
                    else
                    {
                        FatalErrorInFunction << ewhere << ": missing sources or injectionRateSuSp" << exit(FatalError);
                    }
                }
            }
        }
    }
    hSources_.setSize(hList.size());
    forAll(hList, i) hSources_.set(i, hList[i]);
    ySources_.setSize(yList.size());
    forAll(yList, i) ySources_.set(i, yList[i]);
    tConstraints_.setSize(tList.size());
    forAll(tList, i) tConstraints_.set(i, tList[i]);
}

void EnergyCoupling::setupPatches()
{
    std::vector<int> kind(mi_.nB, gc::B_ZEROGRAD);
    std::vector<double> val(mi_.nB, 0.0), grad(mi_.nB, 0.0), vf(mi_.nB, 0.0);
    std::vector<double> Rl(mi_.nIface, 0.0);
    DynamicList<TPatch> list;
    forAll(R_, r)
    {
        const fvMesh& mesh = *R_[r].mesh;
        volScalarField::Boundary& Tbf = T_[r].boundaryFieldRef();
        forAll(Tbf, p)
        {
            const PatchMap& pm = R_[r].patches[p];
            fvPatchScalarField& pf = Tbf[p];
            TPatch tp;
            tp.region = r;
            tp.patch = p;
            tp.type = pf.type();
            tp.n = pf.size();
            const word where = mesh.name() + "/T/" + mesh.boundary()[p].name();
            if (pm.kind == PatchMap::EMPTY)
            {
                tp.kind = TPatch::EMPTY;
                list.append(std::move(tp));
                continue;
            }
            if (pm.kind == PatchMap::IFACE_A || pm.kind == PatchMap::IFACE_B)
            {
                if (!isCoupledT(tp.type))
                    FatalErrorInFunction << where << ": patch is coupled to another region but T has type " << tp.type
                        << exit(FatalError);
                const dictionary d = patchDict(pf);
                const word km = d.getOrDefault<word>("kappaMethod", "fluidThermo");
                if (km != (R_[r].fluid ? "fluidThermo" : "solidThermo"))
                    FatalErrorInFunction << where << ": kappaMethod " << km << " is not supported here"
                        << exit(FatalError);
                for (const word q : {"qr", "qrNbr"})
                    if (d.found(q) && d.get<word>(q) != "none")
                        FatalErrorInFunction << where << ": " << q << " is not supported (none)" << exit(FatalError);
                if (d.getOrDefault<bool>("thermalInertia", false))
                    FatalErrorInFunction << where << ": thermalInertia is not supported" << exit(FatalError);
                if (d.found("thicknessLayers"))
                {
                    const scalarList th(d.get<scalarList>("thicknessLayers"));
                    const scalarList kl(d.get<scalarList>("kappaLayers"));
                    scalar Rs = 0;
                    forAll(th, i) Rs += th[i]/kl[i];
                    const bool rad = tp.type.find("RadCoupled") != std::string::npos;
                    forAll(pm.ifaceIdx, i)
                    {
                        const label j = pm.ifaceIdx[i];
                        Rl[j] = rad ? max(Rl[j], Rs) : Rl[j] + Rs;   // RadCoupled: both sides list the same layers
                    }
                }
                tp.kind = TPatch::COUPLED;
                list.append(std::move(tp));
                continue;
            }
            tp.b0 = pm.b0;
            const word& t = tp.type;
            if (t == "fixedValue" || t == "uniformFixedValue" || t == "timeVaryingMappedFixedValue")
            {
                tp.kind = TPatch::HOSTFIXED;
                for (label i = 0; i < tp.n; ++i) kind[tp.b0 + i] = gc::B_FIXED;
            }
            else if (t == "zeroGradient" || t == "symmetry" || t == "symmetryPlane")
            {
                tp.kind = TPatch::ZEROGRAD;
            }
            else if (t == "fixedGradient")
            {
                tp.kind = TPatch::FIXEDGRAD;
                const scalarField& gr = refCast<fixedGradientFvPatchScalarField>(pf).gradient();
                for (label i = 0; i < tp.n; ++i) { kind[tp.b0 + i] = gc::B_FIXEDGRAD; grad[tp.b0 + i] = gr[i]; }
            }
            else if (t == "mixed")
            {
                tp.kind = TPatch::MIXED;
                const mixedFvPatchScalarField& mp = refCast<mixedFvPatchScalarField>(pf);
                for (label i = 0; i < tp.n; ++i)
                {
                    kind[tp.b0 + i] = gc::B_MIXED;
                    val[tp.b0 + i] = mp.refValue()[i];
                    grad[tp.b0 + i] = mp.refGrad()[i];
                    vf[tp.b0 + i] = mp.valueFraction()[i];
                }
            }
            else if (t == "inletOutlet" || t == "uniformInletOutlet")
            {
                tp.kind = TPatch::INLETOUTLET;
                if (t == "uniformInletOutlet")
                {
                    const dictionary d = patchDict(pf);
                    tp.inletVal = Function1<scalar>::New("uniformInletValue", d, &mesh);
                }
                const mixedFvPatchScalarField& mp = refCast<mixedFvPatchScalarField>(pf);
                for (label i = 0; i < tp.n; ++i)
                {
                    kind[tp.b0 + i] = gc::B_INLETOUTLET;
                    val[tp.b0 + i] = mp.refValue()[i];
                    grad[tp.b0 + i] = 0;
                    vf[tp.b0 + i] = mp.valueFraction()[i];
                }
            }
            else if (t == "externalWallHeatFluxTemperature" || t == "compressible::externalWallHeatFluxTemperature")
            {
                tp.kind = TPatch::EXTWALL;
                const dictionary d = patchDict(pf);
                const word km = d.getOrDefault<word>("kappaMethod", "fluidThermo");
                if (km != (R_[r].fluid ? "fluidThermo" : "solidThermo"))
                    FatalErrorInFunction << where << ": kappaMethod " << km << " is not supported here" << exit(FatalError);
                if (d.found("qr") && d.get<word>("qr") != "none")
                    FatalErrorInFunction << where << ": qr is not supported (none)" << exit(FatalError);
                tp.mode = d.getOrDefault<word>("mode", d.found("h") ? "coefficient" : (d.found("q") ? "flux" : "power"));
                tp.relaxation = d.getOrDefault<scalar>("relaxation", 1);
                const polyPatch& pp = mesh.boundaryMesh()[p];
                if (tp.mode == "power") tp.Q = Function1<scalar>::New("Q", d, &mesh);
                else if (tp.mode == "flux") tp.q = PatchFunction1<scalar>::New(pp, "q", d);
                else if (tp.mode == "coefficient")
                {
                    tp.h = PatchFunction1<scalar>::New(pp, "h", d);
                    tp.Ta = Function1<scalar>::New("Ta", d, &mesh);
                    tp.emissivity = d.getOrDefault<scalar>("emissivity", 0);
                    if (d.found("thicknessLayers"))
                    {
                        const scalarList th(d.get<scalarList>("thicknessLayers"));
                        const scalarList kl(d.get<scalarList>("kappaLayers"));
                        forAll(th, i) if (kl[i] > 0) tp.totalSolidRes += th[i]/kl[i];
                    }
                }
                else
                    FatalErrorInFunction << where << ": mode " << tp.mode << " is not supported" << exit(FatalError);
                const mixedFvPatchScalarField& mp = refCast<mixedFvPatchScalarField>(pf);
                for (label i = 0; i < tp.n; ++i)
                {
                    kind[tp.b0 + i] = gc::B_MIXED;
                    val[tp.b0 + i] = mp.refValue()[i];
                    grad[tp.b0 + i] = mp.refGrad()[i];
                    vf[tp.b0 + i] = mp.valueFraction()[i];
                }
            }
            else if (t == "gpuCht::exteriorSurface")
            {
                if (R_[r].fluid)
                    FatalErrorInFunction << where << ": gpuCht::exteriorSurface is for solid regions"
                        << exit(FatalError);
                tp.kind = TPatch::EXTSURF;
                for (label i = 0; i < tp.n; ++i)
                {
                    kind[tp.b0 + i] = gc::B_MIXED;
                    val[tp.b0 + i] = pf[i];
                    grad[tp.b0 + i] = 0;
                    vf[tp.b0 + i] = 0;
                }
            }
            else
            {
                FatalErrorInFunction << where << ": T boundary type " << t << " is not supported" << exit(FatalError);
            }
            list.append(std::move(tp));
        }
    }
    patches_.transfer(list);
    E_.putBKind(kind);
    E_.put(gc::A::bVal, val);
    E_.put(gc::A::bGrad, grad);
    E_.put(gc::A::bVf, vf);
    E_.put(gc::A::Rlayer, Rl);
    bool anyExt = false;
    for (const TPatch& tp : patches_)
        if (tp.kind == TPatch::EXTSURF) { anyExt = true; break; }
    if (anyExt) setupBuilding();
    updateBCs(true);
}

static std::string resolveCasePath(const Time& runTime, const dictionary& d, const word& key)
{
    ITstream& is = d.lookup(key);
    token tok(is);
    std::string s = tok.isString() ? std::string(tok.stringToken()) : std::string(tok.wordToken());
    auto rep = [](std::string& str, const std::string& from, const std::string& to)
    {
        size_t pos = 0;
        while ((pos = str.find(from, pos)) != std::string::npos)
        {
            str.replace(pos, from.length(), to);
            pos += to.length();
        }
    };
    rep(s, "<constant>", std::string(runTime.constantPath()));
    rep(s, "$FOAM_CASE", std::string(runTime.path()));
    if (!s.empty() && s[0] != '/' && s.find(':') == std::string::npos)
        s = std::string(runTime.path()) + "/" + s;
    return s;
}

void EnergyCoupling::setupBuilding()
{
    IOobject io("gpuChtProperties", runTime_.constant(), runTime_, IOobject::MUST_READ, IOobject::NO_WRITE,
                IOobject::NO_REGISTER);
    if (!io.typeHeaderOk<IOdictionary>(false) && !isFile(io.objectPath()))
        FatalErrorInFunction << "gpuCht::exteriorSurface needs constant/gpuChtProperties" << exit(FatalError);
    IOdictionary d(io);
    gc::BuildingIn bi;
    const dictionary& w = d.subDict("weather");
    bi.weatherFile = resolveCasePath(runTime_, w, "file");
    bi.groundReflectance = w.getOrDefault<scalar>("groundReflectance", 0.2);

    std::set<std::string> glazing = {"glazing_5mm"};
    if (d.isDict("solar"))
    {
        const dictionary& so = d.subDict("solar");
        bi.shading = so.getOrDefault<bool>("shading", true);
        bi.pixel = so.getOrDefault<scalar>("pixel", 0.05);
        bi.solarUpdateInterval = so.getOrDefault<scalar>("updateInterval", 300.0);
        if (so.found("glazing"))
        {
            glazing.clear();
            const wordList gl(so.get<wordList>("glazing"));
            for (const word& g : gl) glazing.insert(g);
        }
        bi.transmittance = so.getOrDefault<scalar>("transmittance", 0.81);
        bi.glassAbsorptance = so.getOrDefault<scalar>("glassAbsorptance", 0.12);
        bi.interiorAbsorptance = so.getOrDefault<scalar>("interiorAbsorptance", 0.7);
    }
    if (d.isDict("interiorRadiation"))
    {
        const dictionary& ir = d.subDict("interiorRadiation");
        bi.interiorRad = ir.getOrDefault<bool>("enabled", true);
        bi.emissivity = ir.getOrDefault<scalar>("emissivity", 0.9);
        bi.glassEmissivity = ir.getOrDefault<scalar>("glassEmissivity", 0.84);
    }

    // exterior faces
    for (const TPatch& tp : patches_)
    {
        if (tp.kind != TPatch::EXTSURF || !tp.n) continue;
        const fvPatchScalarField& pf = T_[tp.region].boundaryField()[tp.patch];
        const exteriorSurfaceFvPatchScalarField& epf = refCast<const exteriorSurfaceFvPatchScalarField>(pf);
        const bool g = glazing.count(R_[tp.region].name) > 0;
        const double hc = epf.hc();
        const double eps = epf.hasEps() ? epf.eps() : (g ? bi.glassEmissivity : 0.9);
        const double alpha = epf.hasAlpha() ? epf.alpha() : (g ? bi.glassAbsorptance : 0.7);
        for (label i = 0; i < tp.n; ++i)
        {
            bi.eb.push_back(tp.b0 + i);
            bi.eHc.push_back(hc);
            bi.eEps.push_back(eps);
            bi.eAlpha.push_back(alpha);
            bi.eGlass.push_back(g ? 1 : 0);
        }
    }

    // per-fluid-region cellZones and cell -> local zone index
    List<List<word>> regZoneNames(R_.size());
    List<labelList> regCellZone(R_.size());
    forAll(R_, r)
    {
        if (!R_[r].fluid) continue;
        const fvMesh& m = *R_[r].mesh;
        const cellZoneMesh& cz = m.cellZones();
        labelList& z = regCellZone[r];
        List<word>& names = regZoneNames[r];
        if (cz.empty())
        {
            names.setSize(1, "air");
            z.setSize(m.nCells(), 0);
        }
        else
        {
            DynamicList<word> nm;
            z.setSize(m.nCells(), -1);
            forAll(cz, k)
            {
                nm.append(cz[k].name());
                const labelList& cells = cz[k];
                forAll(cells, i) z[cells[i]] = k;
            }
            bool anyUnzoned = false;
            forAll(z, c) if (z[c] < 0) { anyUnzoned = true; break; }
            if (anyUnzoned)
            {
                const label uk = nm.size();
                nm.append("_unzoned");
                forAll(z, c) if (z[c] < 0) z[c] = uk;
            }
            names.transfer(nm);
        }
    }

    // interior fluid-solid interfaces
    HashTable<int> zmap;
    DynamicList<word> zoneNames;
    for (const gc::MeshIn::Pair& pr : mi_.pairs)
    {
        const bool flA = R_[pr.rA].fluid, flB = R_[pr.rB].fluid;
        if (flA == flB) continue;
        const label rAir = flA ? pr.rA : pr.rB;
        const label rSol = flA ? pr.rB : pr.rA;
        const List<word>& names = regZoneNames[rAir];
        labelList remap(names.size());
        forAll(names, k)
        {
            if (!zmap.found(names[k]))
            {
                zmap.insert(names[k], zoneNames.size());
                zoneNames.append(names[k]);
            }
            remap[k] = zmap[names[k]];
        }
        const labelList& cz = regCellZone[rAir];
        const bool isGl = glazing.count(R_[rSol].name) > 0;
        const double sgn = flA ? -1.0 : 1.0;
        for (int i = 0; i < pr.n; ++i)
        {
            const int f = pr.f0 + i;
            const int ac = (flA ? mi_.own[f] : mi_.nei[f]) - R_[rAir].c0;
            bi.iPos.push_back(f - nIr_);
            bi.iSign.push_back(sgn);
            bi.iZone.push_back(remap[cz[ac]]);
            bi.iGlass.push_back(isGl ? 1 : 0);
            bi.iAirOwner.push_back(flA ? 1 : 0);
        }
    }
    nZones_ = zoneNames.size();
    bi.nZones = nZones_;

    // interior radiation sources
    DynamicList<RadSource*> rList;
    if (d.isDict("interiorRadiation") && d.subDict("interiorRadiation").isDict("sources"))
    {
        const dictionary& src = d.subDict("interiorRadiation").subDict("sources");
        for (const entry& e : src)
        {
            const word& zn = e.keyword();
            if (zn == "FoamFile" || !zmap.found(zn)) continue;
            RadSource* rs = new RadSource();
            rs->zone = zmap[zn];
            rs->Q = Function1<scalar>::New(zn, src, R_[0].mesh);
            rList.append(rs);
        }
    }
    radSources_.setSize(rList.size());
    forAll(rList, i) radSources_.set(i, rList[i]);

    int nGlaze = 0, nGlazeMiss = 0;
    double areaExt = 0, areaGlaze = 0;
    E_.buildingInit(bi, nGlaze, nGlazeMiss, areaExt, areaGlaze);
    if (nGlaze > 0)
        Info<< "building: " << nGlaze << " glazing faces, " << nGlazeMiss
            << " without a zone behind them (their transmitted light is dropped)" << endl;
    Info<< "building: " << bi.eb.size() << " exterior faces (" << areaExt << " m2, " << areaGlaze
        << " m2 glazing), " << bi.iPos.size() << " interior faces in " << nZones_ << " zones" << endl;
    hasBuilding_ = true;
}

void EnergyCoupling::setupSchemes()
{
    std::vector<double> limK(R_.size()), psi(R_.size());
    forAll(R_, r)
    {
        const fvMesh& mesh = *R_[r].mesh;
        limK[r] = gradLimitK(mesh, "grad(h)");
        psi[r] = laplacianPsi(mesh, R_[r].fluid ? "laplacian(alphaEff,h)" : "laplacian(alpha,h)");
        Info<< "    " << mesh.name() << ": grad(h) limiter " << limK[r] << ", laplacian psi " << psi[r] << endl;
    }
    E_.put(gc::A::rLimitK, limK);
    E_.put(gc::A::rPsi, psi);
    // convection of he: div(phi,h) of the first fluid region (the monolithic system uses one scheme)
    forAll(R_, r)
    {
        if (!R_[r].fluid) continue;
        bool bounded = false;
        scalar k = 1;
        const int conv = divScheme(*R_[r].mesh, "div(phi,h)", "h", 0xF, bounded, k);
        E_.setEnergyConvection(conv, k, bounded);
        forAll(R_, r2)
        {
            if (r2 == r || !R_[r2].fluid) continue;
            bool b2 = false;
            scalar k2 = 1;
            if (divScheme(*R_[r2].mesh, "div(phi,h)", "h", 0xF, b2, k2) != conv || b2 != bounded || k2 != k)
                FatalErrorInFunction << R_[r2].name << ": div(phi,h) differs from region " << R_[r].name
                    << " (one scheme for the monolithic energy system)" << exit(FatalError);
        }
        Info<< "    energy convection div(phi,h): " << conv << (bounded ? " bounded" : "") << endl;
        break;
    }
}

// ----------------------------------------------------------------------------------------- per solve
void EnergyCoupling::updateBCs(bool initCall)
{
    const scalar t = runTime_.timeOutputValue();
    if (!initCall && hasBuilding_)
    {
        std::vector<double> zoneQrad(nZones_, 0.0);
        forAll(radSources_, i)
            zoneQrad[radSources_[i].zone] += radSources_[i].Q->value(t);
        E_.buildingUpdate(t, zoneQrad);
    }
    bool needTb = false;
    for (const TPatch& tp : patches_) needTb = needTb || (tp.kind == TPatch::EXTWALL && tp.n);
    std::vector<double> Tb;
    if (needTb) Tb = E_.boundaryT();
    for (TPatch& tp : patches_)
    {
        if (!tp.n) continue;
        fvPatchScalarField& pf = T_[tp.region].boundaryFieldRef()[tp.patch];
        if (tp.kind == TPatch::HOSTFIXED)
        {
            pf.evaluate();
            E_.put(gc::A::bVal, std::vector<double>(pf.begin(), pf.end()), tp.b0);
        }
        else if (tp.kind == TPatch::INLETOUTLET && tp.inletVal)
        {
            E_.put(gc::A::bVal, std::vector<double>(tp.n, tp.inletVal->value(t)), tp.b0);
        }
        else if (tp.kind == TPatch::EXTWALL)
        {
            // externalWallHeatFluxTemperatureFvPatchScalarField::updateCoeffs (v2606), kappa(Tp) = kappa of the
            // solid (kappaMethod solidThermo)
            const fvPatch& fp = pf.patch();
            const scalarField& dc = fp.deltaCoeffs();
            std::vector<double> kap = E_.get(gc::A::kappaB, tp.b0, tp.n);
            if (R_[tp.region].fluid)
            {
                const std::vector<double> cpb = E_.get(gc::A::cpB, tp.b0, tp.n);
                const std::vector<double> atb = E_.get(gc::A::alphatB, tp.b0, tp.n);
                for (label i = 0; i < tp.n; ++i) kap[i] += cpb[i]*atb[i];
            }
            std::vector<double> rv = E_.get(gc::A::bVal, tp.b0, tp.n);
            std::vector<double> rg = E_.get(gc::A::bGrad, tp.b0, tp.n);
            std::vector<double> vfv = E_.get(gc::A::bVf, tp.b0, tp.n);
            const std::vector<double> rv0(rv), vf0(vfv);
            if (tp.mode == "power")
            {
                const scalar Q = tp.Q->value(t);
                const scalar A = gSum(fp.magSf());
                for (label i = 0; i < tp.n; ++i) { rg[i] = Q/A/kap[i]; rv[i] = 293.15; vfv[i] = 0; }
            }
            else if (tp.mode == "flux")
            {
                const scalarField q(tp.q->value(t));
                for (label i = 0; i < tp.n; ++i) { rg[i] = q[i]/kap[i]; rv[i] = 293.15; vfv[i] = 0; }
            }
            else
            {
                const scalarField htc(tp.h->value(t) + VSMALL);
                const scalar Ta = tp.Ta->value(t);
                const scalar eSig = tp.emissivity*constant::physicoChemical::sigma.value();
                const scalar Rs = tp.totalSolidRes;
                for (label i = 0; i < tp.n; ++i)
                {
                    const scalar Tp = Tb[tp.b0 + i];
                    scalar hrad = 0;
                    if (tp.emissivity > 0)
                    {
                        if (Rs > 0)
                        {
                            const scalar TpLambda = htc[i]/(htc[i] + 1/Rs);
                            scalar Ts = TpLambda*Ta + (1 - TpLambda)*Tp;
                            const scalar h0 = eSig*((pow3(Ta) + pow3(Ts)) + Ta*Ts*(Ta + Ts));
                            scalar TaLambda = (htc[i] + h0)/(htc[i] + h0 + 1/Rs);
                            scalar TsiNew = TaLambda*Ta + (1 - TaLambda)*Tp;
                            scalar Tsi = Ts;
                            while (mag(Tsi - TsiNew)/Tsi > 0.01)
                            {
                                Tsi = TsiNew;
                                const scalar hn = eSig*((pow3(Ta) + pow3(Tsi)) + Ta*Tsi*(Ta + Tsi));
                                TaLambda = (htc[i] + hn)/(htc[i] + hn + 1/Rs);
                                TsiNew = TaLambda*Ta + (1 - TaLambda)*Tp;
                            }
                            hrad = eSig*((pow3(Ta) + pow3(Tsi)) + Ta*Tsi*(Ta + Tsi));
                        }
                        else
                        {
                            hrad = eSig*((pow3(Ta) + pow3(Tp)) + Ta*Tp*(Ta + Tp));
                        }
                    }
                    const scalar hp = 1/(1/(htc[i] + hrad) + Rs);
                    rg[i] = 0;
                    rv[i] = hp*Ta/hp;
                    vfv[i] = hp/(hp + kap[i]*dc[i]);
                }
            }
            const scalar a = tp.relaxation;
            for (label i = 0; i < tp.n; ++i)
            {
                vfv[i] = vf0[i] + a*(vfv[i] - vf0[i]);
                rv[i] = rv0[i] + a*(rv[i] - rv0[i]);
            }
            E_.put(gc::A::bVal, rv, tp.b0);
            E_.put(gc::A::bGrad, rg, tp.b0);
            E_.put(gc::A::bVf, vfv, tp.b0);
        }
    }
    if (!hSources_.empty())
    {
        std::vector<double> Su(mi_.nC, 0.0), Sp(mi_.nC, 0.0);
        forAll(hSources_, k)
        {
            const SemiImplicitSource& s = hSources_[k];
            const scalar su = s.SuF ? s.SuF->value(t) : s.SuConst;
            const scalar sp = s.SpF ? s.SpF->value(t) : s.SpConst;
            for (size_t i = 0; i < s.cells.size(); ++i)
            {
                Su[s.cells[i]] += su*s.wV[i];
                Sp[s.cells[i]] += sp*s.wV[i];
            }
        }
        E_.put(gc::A::Su, Su);
        E_.put(gc::A::Sp, Sp);
    }
    if (!tConstraints_.empty())
    {
        std::vector<int> cells;
        std::vector<double> vals;
        forAll(tConstraints_, k)
        {
            const FixedTempConstraint& ft = tConstraints_[k];
            const double tv = ft.Tf->value(t);
            cells.insert(cells.end(), ft.gcells.begin(), ft.gcells.end());
            vals.insert(vals.end(), ft.gcells.size(), tv);
        }
        E_.setFixedT(cells, vals);
    }
}

void EnergyCoupling::updateSpecieSources(label r, int fluidId) const
{
    if (ySources_.empty()) return;
    const scalar t = runTime_.timeOutputValue();
    const FluidProps& fp = fluid_[r];
    const label nc = R_[r].mesh->nCells();
    forAll(fp.species, m)
    {
        if (m == fp.inertIndex) continue;
        bool any = false;
        forAll(ySources_, k)
            if (ySources_[k].region == r && ySources_[k].specie == m) { any = true; break; }
        if (!any) continue;
        std::vector<double> Su(nc, 0.0), Sp(nc, 0.0);
        forAll(ySources_, k)
        {
            const SemiImplicitSource& s = ySources_[k];
            if (s.region != r || s.specie != m) continue;
            const scalar su = s.SuF ? s.SuF->value(t) : s.SuConst;
            const scalar sp = s.SpF ? s.SpF->value(t) : s.SpConst;
            for (size_t i = 0; i < s.cells.size(); ++i)
            {
                Su[s.cells[i]] += su*s.wV[i];
                Sp[s.cells[i]] += sp*s.wV[i];
            }
        }
        E_.fputSpecieSource(fluidId, m, Su, Sp);
    }
}

void EnergyCoupling::setRelax(bool finalIter)
{
    std::vector<double> rr(R_.size(), 0.0);
    forAll(R_, r)
    {
        const word name = (R_[r].fluid && finalIter) ? "hFinal" : "h";
        scalar f = 0;
        if (R_[r].mesh->relaxEquation(name, f)) rr[r] = f;
    }
    E_.put(gc::A::rRelax, rr);
}

gc::SolverCtl EnergyCoupling::solverCtl(bool finalIter) const
{
    // one monolithic solve: settings of the first fluid region's h (else the first solid's)
    label r0 = 0;
    forAll(R_, r) if (R_[r].fluid) { r0 = r; break; }
    return readSolver(*R_[r0].mesh, finalIter ? "hFinal" : "h", false);
}

void EnergyCoupling::logMinMax() const
{
    forAll(R_, r)
    {
        double mn, mx;
        E_.minMax(gc::A::T, R_[r].c0, R_[r].mesh->nCells(), mn, mx);
        Info<< "Min/max T:" << mn << ' ' << mx << "  (" << R_[r].name << ")" << endl;
    }
}

scalar EnergyCoupling::solidDiffNo(scalar dt, scalar& meanDi) const
{
    scalar DiNum = -GREAT;
    meanDi = 0.0;
    forAll(R_, r)
    {
        if (R_[r].fluid) continue;
        const fvMesh& mesh = *R_[r].mesh;
        const SolidProps& s = solid_[r];
        const dimensionedScalar alpha("alpha", sqr(dimLength)/dimTime, s.kappa/(s.rho*s.Cp));
        const surfaceScalarField kapparhoCpbyDelta
        (
            sqr(mesh.surfaceInterpolation::deltaCoeffs())*alpha
        );
        const scalar diR = max(kapparhoCpbyDelta).value()*dt;
        const scalar meanDiR = average(kapparhoCpbyDelta).value()*dt;
        Info<< "Region: " << mesh.name() << " Diffusion Number mean: " << meanDiR
            << " max: " << diR << endl;
        DiNum = max(DiNum, diR);
        meanDi = max(meanDi, meanDiR);
    }
    return DiNum;
}

// ----------------------------------------------------------------------------------------- output
void EnergyCoupling::writeBack()
{
    const std::vector<double> Tc = E_.get(gc::A::T);
    const std::vector<double> Tb = E_.boundaryT();
    std::vector<double> TA, TB, gA, gB;
    E_.interfaceState(TA, TB, gA, gB);
    const std::vector<double> bv = E_.get(gc::A::bVal), bg = E_.get(gc::A::bGrad), bf = E_.get(gc::A::bVf);
    forAll(R_, r)
    {
        scalarField& Ti = T_[r].primitiveFieldRef();
        forAll(Ti, c) Ti[c] = Tc[R_[r].c0 + c];
    }
    for (const TPatch& tp : patches_)
    {
        if (tp.kind == TPatch::EMPTY || !tp.n) continue;
        fvPatchScalarField& pf = T_[tp.region].boundaryFieldRef()[tp.patch];
        scalarField v(tp.n);
        if (tp.kind == TPatch::COUPLED)
        {
            const PatchMap& pm = R_[tp.region].patches[tp.patch];
            const bool A = pm.kind == PatchMap::IFACE_A;
            mixedFvPatchScalarField& mp = refCast<mixedFvPatchScalarField>(pf);
            forAll(v, i)
            {
                const label j = pm.ifaceIdx[i];
                const label f = pm.gface[i];
                v[i] = A ? TA[j] : TB[j];
                const label nbrCell = A ? mi_.nei[f] : mi_.own[f];
                const scalar mine = A ? gA[j] : gB[j], nbr = A ? gB[j] : gA[j];
                mp.refValue()[i] = Tc[nbrCell];
                mp.refGrad()[i] = 0;
                mp.valueFraction()[i] = nbr/(nbr + mine);
            }
        }
        else
        {
            forAll(v, i) v[i] = Tb[tp.b0 + i];
            if (tp.kind == TPatch::MIXED || tp.kind == TPatch::INLETOUTLET || tp.kind == TPatch::EXTWALL
             || tp.kind == TPatch::EXTSURF)
            {
                mixedFvPatchScalarField& mp = refCast<mixedFvPatchScalarField>(pf);
                forAll(v, i)
                {
                    mp.refValue()[i] = bv[tp.b0 + i];
                    mp.refGrad()[i] = bg[tp.b0 + i];
                    mp.valueFraction()[i] = bf[tp.b0 + i];
                }
            }
        }
        pf == v;
    }
}

} // namespace gpuCht
} // namespace Foam
