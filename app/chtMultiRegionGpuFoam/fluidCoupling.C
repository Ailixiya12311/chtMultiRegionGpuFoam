#include "fluidCoupling.H"

#include "findRefCell.H"
#include "mixedFvPatchFields.H"
#include "fixedGradientFvPatchFields.H"
#include "calculatedFvPatchFields.H"
#include "IStringStream.H"

#include <algorithm>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>

namespace Foam
{
namespace gpuCht
{

// equation / field relaxation factor as fvMatrix::relax() and GeometricField::relax() of v2606 look it up:
// name = finalIter ? "<name>Final" : "<name>" (no fall-back), solution::relaxEquation / relaxField(name, factor);
// <= 0: not found (fvMatrix: no relaxation; field: relax(1) is a no-op)
static scalar relaxFactor(const fvMesh& mesh, const word& name, bool finalIter, bool field)
{
    const word n = finalIter ? word(name + "Final") : name;
    scalar f = 0;
    if (field ? mesh.relaxField(n, f) : mesh.relaxEquation(n, f)) return f;
    return 0;
}

static void logSolve(const word& solver, const word& field, const gc::SolveInfo& si)
{
    Info<< solver.c_str() << ":  Solving for " << field << ", Initial residual = " << si.res0
        << ", Final residual = " << si.res << ", No Iterations " << si.iters << endl;
}

// ----------------------------------------------------------------------------------------- setup
FluidCoupling::FluidCoupling(Time& runTime, List<Region>& regions, const gc::MeshIn& mi, gc::Engine& E,
                             EnergyCoupling& energy, label nOuterCorr)
:
    runTime_(runTime),
    R_(regions),
    mi_(mi),
    E_(E),
    energy_(energy),
    nOuterCorr_(nOuterCorr)
{
    label n = 0;
    forAll(R_, r) if (R_[r].fluid) ++n;
    F_.setSize(n);
    n = 0;
    forAll(R_, r)
    {
        if (!R_[r].fluid) continue;
        F_.set(n, new FluidRegion());
        F_[n].r = r;
        setup(F_[n]);
        ++n;
    }
    setupHvac();
}

void FluidCoupling::setup(FluidRegion& f)
{
    const Region& R = R_[f.r];
    fvMesh& mesh = *R.mesh;
    const word tn = runTime_.timeName();
    const FluidProps& fp = energy_.fluidProps(f.r);
    const word where = mesh.name();

    // ---- fields (createFluidFields.H)
    f.U.reset(new volVectorField(IOobject("U", tn, mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE), mesh));
    f.p.reset(new volScalarField(IOobject("p", tn, mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE), mesh));
    f.p_rgh.reset(new volScalarField(IOobject("p_rgh", tn, mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE), mesh));
    f.rho.reset(new volScalarField(IOobject("rho", tn, mesh, IOobject::NO_READ, IOobject::AUTO_WRITE), mesh,
                                   dimensionedScalar(dimDensity, Zero), calculatedFvPatchScalarField::typeName));
    IOobject phio("phi", tn, mesh, IOobject::READ_IF_PRESENT, IOobject::AUTO_WRITE);
    const bool phiGiven = phio.typeHeaderOk<surfaceScalarField>(true);
    if (phiGiven)
    {
        Info<< "    " << where << ": reading phi" << endl;
        f.phi.reset(new surfaceScalarField(phio, mesh));
    }
    else
    {
        f.phi.reset(new surfaceScalarField(IOobject("phi", tn, mesh, IOobject::NO_READ, IOobject::AUTO_WRITE), mesh,
                                           dimensionedScalar(dimMass/dimTime, Zero)));
    }
    const volVectorField& U = f.U();
    const volScalarField& p = f.p();
    volScalarField& p_rgh = f.p_rgh();
    if (fp.multiComponent)
    {
        const label nSp = fp.species.size();
        f.Y.setSize(nSp);
        forAll(fp.species, m)
        {
            f.Y.set
            (
                m,
                new volScalarField
                (
                    IOobject(fp.species[m].name, tn, mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE),
                    mesh
                )
            );
        }
        // multiComponentMixture::correctMassFractions()
        volScalarField Yt("Yt", 0.0*f.Y[0]);
        forAll(f.Y, m) Yt += f.Y[m];
        forAll(f.Y, m) f.Y[m] /= Yt;
    }
    f.turbulent = fp.turbulent;
    if (f.turbulent)
    {
        f.k.reset(new volScalarField(IOobject("k", tn, mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE), mesh));
        f.epsilon.reset
        (
            new volScalarField(IOobject("epsilon", tn, mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE), mesh)
        );
        f.nut.reset(new volScalarField(IOobject("nut", tn, mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE), mesh));
        f.alphat.reset
        (
            new volScalarField(IOobject("alphat", tn, mesh, IOobject::MUST_READ, IOobject::AUTO_WRITE), mesh)
        );
    }

    // ---- gravity (per region or global constant/) and hRef
    const objectRegistry& gDb = isFile(runTime_.constantPath()/mesh.dbDir()/"g")
        ? static_cast<const objectRegistry&>(mesh)
        : static_cast<const objectRegistry&>(runTime_);
    uniformDimensionedVectorField g(IOobject("g", runTime_.constant(), gDb, IOobject::MUST_READ, IOobject::NO_WRITE));

    const objectRegistry& hRefDb = isFile(runTime_.constantPath()/mesh.dbDir()/"hRef")
        ? static_cast<const objectRegistry&>(mesh)
        : static_cast<const objectRegistry&>(runTime_);
    uniformDimensionedScalarField hRef
    (
        IOobject("hRef", runTime_.constant(), hRefDb, IOobject::READ_IF_PRESENT, IOobject::NO_WRITE),
        dimensionedScalar(dimLength, Zero)
    );
    // createFluidFields.H: ghRef = mag(g) > SMALL ? g & (cmptMag(g)/mag(g))*hRef : 0
    const vector gv = g.value();
    const scalar ghRef = mag(gv) > SMALL ? (gv & (cmptMag(gv)/mag(gv)))*hRef.value() : 0;

    // ---- slots: region boundary faces, then the interface faces patch by patch
    gc::FluidIn fi;
    fi.region = f.r;
    fi.Cp = fp.Cp;
    fi.Hsref = fp.Hsref;
    fi.Tref = fp.Tref;
    fi.kappa = fp.kappa;
    f.nBr = mi_.regions[f.r].nB;
    f.slot0.setSize(mesh.boundary().size(), -1);
    label s = f.nBr;
    forAll(mesh.boundary(), pi)
    {
        const PatchMap& pm = R.patches[pi];
        if (pm.kind == PatchMap::BOUNDARY)
        {
            f.slot0[pi] = pm.b0 - R.b0;
        }
        else if (pm.kind == PatchMap::IFACE_A || pm.kind == PatchMap::IFACE_B)
        {
            const gc::MeshIn::Pair& P = mi_.pairs[pm.pair];
            if (R_[P.rA].fluid && R_[P.rB].fluid)
                FatalErrorInFunction << where << "/" << mesh.boundary()[pi].name()
                    << ": interfaces between two fluid regions are not supported" << exit(FatalError);
            f.slot0[pi] = s;
            forAll(pm.gface, i)
            {
                fi.ifFaces.push_back(pm.gface[i]);
                fi.ifSideA.push_back(pm.kind == PatchMap::IFACE_A ? 1 : 0);
            }
            s += pm.gface.size();
        }
    }
    f.nS = s;
    for (int k = 0; k < 3; ++k) fi.g[k] = g.value()[k];
    fi.ghRef = ghRef;
    const Vector<label> sd = mesh.solutionD();
    for (int k = 0; k < 3; ++k) fi.solveDir[k] = sd[k] > 0;

    // ---- schemes
    bool bnd = false;
    scalar coef = 1;
    fi.conv = divScheme(mesh, "div(phi,U)", "U", 0x3, bnd, coef);
    fi.bounded = bnd;
    fi.limitKU = gradLimitK(mesh, "grad(U)");
    fi.psiU = laplacianPsi(mesh, "laplacian((rho*nuEff),U)");
    fi.psiP = laplacianPsi(mesh, "laplacian(rhorAUf,p_rgh)");
    fi.psiSnP = snGradPsi(mesh, "snGrad(p_rgh)");
    fi.psiSnRho = snGradPsi(mesh, "snGrad(rho)");
    for (const word gname : {"grad(p_rgh)", "grad(rho)"})
        if (gradLimitK(mesh, gname) != 0)
            FatalErrorInFunction << where << ": " << gname << " must be leastSquares (no limiter)" << exit(FatalError);
    for (const word dname : {"div(((rho*nuEff)*dev2(T(grad(U)))))", "div(phi,K)"})
    {
        if (divScheme(mesh, dname, "", 1 << 2, bnd, coef) != 2 || bnd)
            FatalErrorInFunction << where << ": " << dname << " must be Gauss linear" << exit(FatalError);
    }
    if (fp.multiComponent)
    {
        fi.inertIndex = fp.inertIndex;
        fi.convY = divScheme(mesh, "div(phi,Yi_h)", "Yi_h", 0xF, bnd, coef);
        fi.convYK = coef;
        fi.boundedY = bnd;
        fi.limitKH = gradLimitK(mesh, "grad(h)");
        fi.species.resize(fp.species.size());
        forAll(fp.species, m)
        {
            const SpecieProps& sp = fp.species[m];
            gc::SpecieIn& gs = fi.species[m];
            gs.name = sp.name;
            gs.Cp = sp.Cp;
            gs.Hsref = sp.Hsref;
            gs.Tref = sp.Tref;
            gs.mu = sp.mu;
            gs.Pr = sp.Pr;
            gs.rho0 = sp.rho0;
            gs.T0 = sp.T0;
            gs.beta = sp.beta;
            gs.limitK = gradLimitK(mesh, "grad(" + sp.name + ")");
            gs.psiY = laplacianPsi(mesh, "laplacian(muEff," + sp.name + ")");
        }
    }
    if (f.turbulent)
    {
        fi.turb = fp.turb;
        fi.turb.convK = divScheme(mesh, "div(phi,k)", "k", 0xF, bnd, coef);
        fi.turb.boundedK = bnd;
        fi.turb.convKK = coef;
        fi.turb.convEps = divScheme(mesh, "div(phi,epsilon)", "epsilon", 0xF, bnd, coef);
        fi.turb.boundedEps = bnd;
        fi.turb.convEpsK = coef;
        fi.turb.limitKK = gradLimitK(mesh, "grad(k)");
        fi.turb.limitKEps = gradLimitK(mesh, "grad(epsilon)");
        fi.turb.psiK = laplacianPsi(mesh, "laplacian((rho*DkEff),k)");
        fi.turb.psiEps = laplacianPsi(mesh, "laplacian((rho*DepsilonEff),epsilon)");
    }

    // ---- PIMPLE controls of the region
    const dictionary& pd = mesh.solutionDict().subDict("PIMPLE");
    f.momentumPredictor = pd.getOrDefault<bool>("momentumPredictor", true);
    f.nCorr = pd.getOrDefault<label>("nCorrectors", 1);
    f.nNonOrth = pd.getOrDefault<label>("nNonOrthogonalCorrectors", 0);
    f.frozenFlow = pd.getOrDefault<bool>("frozenFlow", false);
    if (pd.found("rhoMin") || pd.found("rhoMax"))
        FatalErrorInFunction << where << ": rhoMin / rhoMax are not implemented yet" << exit(FatalError);
    if (pd.getOrDefault<bool>("consistent", false))
        FatalErrorInFunction << where << ": consistent (SIMPLEC) is not supported" << exit(FatalError);
    fi.pRefCell = -1;
    fi.pRefValue = 0;
    if (p_rgh.needReference())
    {
        label refCell = -1;
        scalar refValue = 0;
        setRefCell(p, p_rgh, pd, refCell, refValue);
        fi.pRefCell = refCell;
        fi.pRefValue = refValue;
        Info<< "    " << where << ": p reference cell " << refCell << " value " << refValue << endl;
    }

    f.id = E_.addFluid(fi);
    if (E_.nSlots(f.id) != f.nS)
        FatalErrorInFunction << where << ": slot count mismatch" << exit(FatalError);

    const label nc = mesh.nCells();
    const label nS = f.nS;
    if (phiGiven)
    {
        const surfaceScalarField& phi = f.phi();
        E_.fput(f.id, gc::F::phi, std::vector<double>(phi.primitiveField().begin(), phi.primitiveField().end()));
        std::vector<double> phiB(nS, 0.0);
        forAll(mesh.boundary(), pi)
        {
            if (f.slot0[pi] < 0) continue;
            const fvsPatchScalarField& pp = phi.boundaryField()[pi];
            forAll(pp, i) phiB[f.slot0[pi] + i] = pp[i];
        }
        E_.fput(f.id, gc::F::phiB, phiB);
    }

    // ---- species & mixture properties (or pureMixture constants)
    DynamicList<ScaPatchF1*> yValList;
    if (fp.multiComponent)
    {
        forAll(f.Y, m)
        {
            const volScalarField& Ym = f.Y[m];
            std::vector<double> yC(Ym.primitiveField().begin(), Ym.primitiveField().end());
            std::vector<int> yK(nS, gc::B_ZEROGRAD);
            std::vector<double> yVal(nS, 0.0), yGrad(nS, 0.0), yVf(nS, 0.0);
            forAll(mesh.boundary(), pi)
            {
                const label s0 = f.slot0[pi];
                if (s0 < 0) continue;
                const fvPatchScalarField& Yp = Ym.boundaryField()[pi];
                const word& yt = Yp.type();
                const word pw = where + "/" + Ym.name() + "/" + mesh.boundary()[pi].name();
                if ((yt == "uniformFixedValue" || yt == "uniformInletOutlet") && Yp.size() > 0)
                {
                    const dictionary yd = patchDict(Yp);
                    ScaPatchF1* yf1 = new ScaPatchF1();
                    yf1->s0 = s0;
                    yf1->n = Yp.size();
                    yf1->specie = m;
                    yf1->f1 = Function1<scalar>::New
                    (
                        yt == "uniformFixedValue" ? "uniformValue" : "uniformInletValue",
                        yd,
                        &mesh
                    );
                    yValList.append(yf1);
                }
                forAll(Yp, i)
                {
                    const label sl = s0 + i;
                    if (yt == "fixedValue" || yt == "uniformFixedValue")
                    {
                        yK[sl] = gc::B_FIXED;
                        yVal[sl] = Yp[i];
                    }
                    else if (yt == "zeroGradient" || yt == "symmetry" || yt == "symmetryPlane" || yt == "calculated")
                    {
                        yK[sl] = gc::B_ZEROGRAD;
                    }
                    else if (yt == "fixedGradient")
                    {
                        yK[sl] = gc::B_FIXEDGRAD;
                        yGrad[sl] = refCast<const fixedGradientFvPatchScalarField>(Yp).gradient()[i];
                    }
                    else if (yt == "mixed")
                    {
                        const mixedFvPatchScalarField& mp = refCast<const mixedFvPatchScalarField>(Yp);
                        yK[sl] = gc::B_MIXED;
                        yVal[sl] = mp.refValue()[i];
                        yGrad[sl] = mp.refGrad()[i];
                        yVf[sl] = mp.valueFraction()[i];
                    }
                    else if (yt == "inletOutlet" || yt == "uniformInletOutlet")
                    {
                        const mixedFvPatchScalarField& mp = refCast<const mixedFvPatchScalarField>(Yp);
                        yK[sl] = gc::B_INLETOUTLET;
                        yVal[sl] = mp.refValue()[i];
                        yVf[sl] = mp.valueFraction()[i];
                    }
                    else
                    {
                        FatalErrorInFunction << pw << ": specie boundary type " << yt << " is not supported"
                            << exit(FatalError);
                    }
                }
            }
            E_.fputSpecie(f.id, m, yC, yK, yVal, yGrad, yVf);
        }
        E_.fluidUpdateMixture(f.id);
    }
    else
    {
        E_.fput(f.id, gc::F::mu, std::vector<double>(nc, fp.mu));
        E_.fput(f.id, gc::F::muB, std::vector<double>(nS, fp.mu));
        E_.fput(f.id, gc::F::eRho0, std::vector<double>(nc, fp.rho0));
        E_.fput(f.id, gc::F::eT0, std::vector<double>(nc, fp.T0));
        E_.fput(f.id, gc::F::eBeta, std::vector<double>(nc, fp.beta));
        E_.fput(f.id, gc::F::eRho0B, std::vector<double>(nS, fp.rho0));
        E_.fput(f.id, gc::F::eT0B, std::vector<double>(nS, fp.T0));
        E_.fput(f.id, gc::F::eBetaB, std::vector<double>(nS, fp.beta));
    }
    E_.fluidUpdateRho(f.id);
    const std::vector<double> rhoC = E_.fget(f.id, gc::F::rho);
    const std::vector<double> rhoB = E_.fget(f.id, gc::F::rhoB);

    // ---- U and p_rgh boundary conditions
    std::vector<int> uK(nS, gc::UK_FIXED), pK(nS, gc::PK_ZEROGRAD);
    std::vector<char> uA(nS, 0), uF(nS, 1);
    std::vector<double> uVal(3*nS, 0.0), pVal(nS, 0.0), pGrad(nS, 0.0), p0Val(nS, 0.0);
    std::vector<gc::FlowPatchIn> fpList;
    DynamicList<FlowPatchF1*> uFlowList;
    DynamicList<VecPatchF1*> uValList;
    DynamicList<ScaPatchF1*> pValList;
    forAll(mesh.boundary(), pi)
    {
        const label s0 = f.slot0[pi];
        if (s0 < 0) continue;
        const word pw = where + "/" + mesh.boundary()[pi].name();
        const fvPatchVectorField& Up = U.boundaryField()[pi];
        const word& ut = Up.type();
        int k = -1;
        if (ut == "noSlip" || ut == "fixedValue" || ut == "movingWallVelocity" || ut == "uniformFixedValue"
         || ut == "flowRateInletVelocity" || ut == "flowRateOutletVelocity")
        {
            k = gc::UK_FIXED;
        }
        else if (ut == "zeroGradient") k = gc::UK_ZEROGRAD;
        else if (ut == "inletOutlet" || ut == "uniformInletOutlet") k = gc::UK_INLETOUTLET;
        else if (ut == "pressureInletOutletVelocity")
        {
            if (patchDict(Up).found("tangentialVelocity"))
                FatalErrorInFunction << pw << ": tangentialVelocity is not supported" << exit(FatalError);
            k = gc::UK_PIOV;
        }
        else if (ut == "slip" || ut == "symmetry" || ut == "symmetryPlane") k = gc::UK_SLIP;
        else
            FatalErrorInFunction << pw << ": U boundary type " << ut << " is not supported" << exit(FatalError);

        if ((ut == "uniformFixedValue" || ut == "uniformInletOutlet") && Up.size() > 0)
        {
            const dictionary ud = patchDict(Up);
            VecPatchF1* vf1 = new VecPatchF1();
            vf1->s0 = s0;
            vf1->n = Up.size();
            vf1->f1 = Function1<vector>::New
            (
                ut == "uniformFixedValue" ? "uniformValue" : "uniformInletValue",
                ud,
                &mesh
            );
            uValList.append(vf1);
        }
        else if ((ut == "flowRateInletVelocity" || ut == "flowRateOutletVelocity") && Up.size() > 0)
        {
            const dictionary ud = patchDict(Up);
            if (ud.getOrDefault<bool>("extrapolateProfile", false))
                FatalErrorInFunction << pw << ": extrapolateProfile is not supported" << exit(FatalError);
            gc::FlowPatchIn fpIn;
            fpIn.s0 = s0;
            fpIn.n = Up.size();
            fpIn.outlet = (ut == "flowRateOutletVelocity");
            FlowPatchF1* ff1 = new FlowPatchF1();
            ff1->fpIndex = int(fpList.size());
            ff1->patch = pi;
            if (ud.found("volumetricFlowRate"))
            {
                fpIn.mass = false;
                ff1->q = Function1<scalar>::New("volumetricFlowRate", ud, &mesh);
            }
            else if (ud.found("massFlowRate"))
            {
                fpIn.mass = true;
                ff1->q = Function1<scalar>::New("massFlowRate", ud, &mesh);
            }
            else
            {
                FatalErrorInFunction << pw << ": " << ut << " needs volumetricFlowRate or massFlowRate"
                    << exit(FatalError);
            }
            fpIn.flowRate = ff1->q->value(runTime_.timeOutputValue());
            fpList.push_back(fpIn);
            uFlowList.append(ff1);
        }

        const bool iface = R.patches[pi].kind != PatchMap::BOUNDARY;
        if (iface && !(Up.fixesValue() && !Up.assignable()))
            FatalErrorInFunction << pw << ": U on a region interface must be a wall (noSlip / fixedValue)"
                << exit(FatalError);
        const fvPatchScalarField& Pp = p_rgh.boundaryField()[pi];
        const word& pt = Pp.type();
        int pk = -1;
        scalarField p0Field;
        if (pt == "fixedFluxPressure") pk = gc::PK_FLUX;
        else if (pt == "zeroGradient" || pt == "symmetry" || pt == "symmetryPlane") pk = gc::PK_ZEROGRAD;
        else if (pt == "fixedValue" || pt == "uniformFixedValue")
        {
            pk = gc::PK_FIXED;
            if (pt == "uniformFixedValue" && Pp.size() > 0)
            {
                const dictionary pd_bc = patchDict(Pp);
                ScaPatchF1* pf1 = new ScaPatchF1();
                pf1->s0 = s0;
                pf1->n = Pp.size();
                pf1->f1 = Function1<scalar>::New("uniformValue", pd_bc, &mesh);
                pValList.append(pf1);
            }
        }
        else if (pt == "prghPressure")
        {
            pk = gc::PK_PRGH_P;
            const dictionary pd_bc = patchDict(Pp);
            p0Field = scalarField("p", pd_bc, Pp.size());
        }
        else if (pt == "prghTotalPressure")
        {
            pk = gc::PK_PRGH_TOTALP;
            const dictionary pd_bc = patchDict(Pp);
            p0Field = scalarField("p0", pd_bc, Pp.size());
        }
        else
            FatalErrorInFunction << pw << ": p_rgh boundary type " << pt << " is not supported" << exit(FatalError);
        const scalarField& pb = p.boundaryField()[pi];
        const vectorField& Cf = mesh.boundary()[pi].Cf();
        forAll(Up, i)
        {
            const label sl = s0 + i;
            uK[sl] = k;
            uA[sl] = Up.assignable();
            uF[sl] = Up.fixesValue();
            vector v = Zero;
            if (k == gc::UK_FIXED) v = Up[i];
            else if (k == gc::UK_INLETOUTLET) v = refCast<const mixedFvPatchVectorField>(Up).refValue()[i];
            uVal[3*sl] = v.x();
            uVal[3*sl + 1] = v.y();
            uVal[3*sl + 2] = v.z();
            pK[sl] = pk;
            if (pk == gc::PK_FLUX) pGrad[sl] = refCast<const fixedGradientFvPatchScalarField>(Pp).gradient()[i];
            // createFluidFields: p_rgh = p - rho gh (fixed-value patches too)
            if (pk == gc::PK_FIXED) pVal[sl] = pb[i] - rhoB[sl]*((g.value() & Cf[i]) - ghRef);
            else if (pk == gc::PK_PRGH_P || pk == gc::PK_PRGH_TOTALP)
            {
                p0Val[sl] = p0Field[i];
                pVal[sl] = p0Field[i] - rhoB[sl]*((g.value() & Cf[i]) - ghRef);
            }
        }
    }
    f.uFlowF1.setSize(uFlowList.size());
    forAll(uFlowList, i) f.uFlowF1.set(i, uFlowList[i]);
    f.uValF1.setSize(uValList.size());
    forAll(uValList, i) f.uValF1.set(i, uValList[i]);
    f.pValF1.setSize(pValList.size());
    forAll(pValList, i) f.pValF1.set(i, pValList[i]);
    f.yValF1.setSize(yValList.size());
    forAll(yValList, i) f.yValF1.set(i, yValList[i]);

    E_.fputKinds(f.id, uK, pK, uA, uF);
    E_.fput(f.id, gc::F::uVal, uVal);
    E_.fput(f.id, gc::F::pVal, pVal);
    E_.fput(f.id, gc::F::pGrad, pGrad);
    E_.fput(f.id, gc::F::p0Val, p0Val);
    if (!fpList.empty()) E_.fputFlowPatches(f.id, fpList);

    // ---- cell fields
    std::vector<double> Uc(3*nc), pc(nc);
    forAll(U, c)
    {
        Uc[3*c] = U[c].x();
        Uc[3*c + 1] = U[c].y();
        Uc[3*c + 2] = U[c].z();
        const scalar gh = (g.value() & mesh.C()[c]) - ghRef;
        pc[c] = p[c] - rhoC[c]*gh;
        p_rgh.primitiveFieldRef()[c] = pc[c];
    }
    E_.fput(f.id, gc::F::U, Uc);
    E_.fput(f.id, gc::F::p, pc);

    if (f.turbulent)
    {
        const volScalarField& k = f.k();
        const volScalarField& eps = f.epsilon();
        const volScalarField& nut = f.nut();
        const volScalarField& alphat = f.alphat();
        std::vector<double> kc(k.primitiveField().begin(), k.primitiveField().end());
        std::vector<double> ec(eps.primitiveField().begin(), eps.primitiveField().end());
        std::vector<double> nc_v(nut.primitiveField().begin(), nut.primitiveField().end());
        std::vector<double> ac(alphat.primitiveField().begin(), alphat.primitiveField().end());
        std::vector<int> kK(nS, gc::B_ZEROGRAD), eK(nS, gc::B_ZEROGRAD);
        std::vector<int> nK(nS, gc::NUT_CALC), aK(nS, gc::AT_CALC);
        std::vector<double> kV(nS, 0.0), eV(nS, 0.0), kb0(nS, 0.0), epsb0(nS, 0.0);
        std::vector<double> eCmu(nS, 0.09), eKappa(nS, 0.41);
        std::vector<double> nCmu(nS, 0.09), nKappa(nS, 0.41), nE(nS, 9.8);
        std::vector<double> atPrt(nS, 0.85), atKappa(nS, 0.41), atE(nS, 9.8);
        std::vector<double> sY(nS, 1.0), alphatB0(nS, 0.0);
        forAll(mesh.boundary(), pi)
        {
            const label s0 = f.slot0[pi];
            if (s0 < 0) continue;
            const word pName = mesh.boundary()[pi].name();
            const scalarField& dc = mesh.boundary()[pi].deltaCoeffs();
            const fvPatchScalarField& kp = k.boundaryField()[pi];
            const fvPatchScalarField& ep = eps.boundaryField()[pi];
            const fvPatchScalarField& np = nut.boundaryField()[pi];
            const fvPatchScalarField& ap = alphat.boundaryField()[pi];
            const word& kt = kp.type();
            const word& et = ep.type();
            const word& nt = np.type();
            const word& at = ap.type();
            const dictionary kd = patchDict(kp);
            const dictionary ed = patchDict(ep);
            const dictionary nd = patchDict(np);
            const dictionary ad = patchDict(ap);
            int kKindP = gc::B_ZEROGRAD;
            scalar kIntensity = 0;
            if (kt == "fixedValue" || kt == "uniformFixedValue") kKindP = gc::B_FIXED;
            else if (kt == "kqRWallFunction" || kt == "zeroGradient" || kt == "symmetry" || kt == "symmetryPlane"
                  || kt == "calculated")
                kKindP = gc::B_ZEROGRAD;
            else if (kt == "inletOutlet" || kt == "uniformInletOutlet") kKindP = gc::B_INLETOUTLET;
            else if (kt == "turbulentIntensityKineticEnergyInlet")
            {
                if (kd.getOrDefault<word>("U", "U") != "U" || kd.getOrDefault<word>("phi", "phi") != "phi")
                    FatalErrorInFunction << where << "/k/" << pName << ": non-default U/phi is not supported"
                        << exit(FatalError);
                kKindP = gc::B_TURB_INTENSITY;
                kIntensity = kd.get<scalar>("intensity");
            }
            else
                FatalErrorInFunction << where << "/k/" << pName << ": k boundary type " << kt << " is not supported"
                    << exit(FatalError);

            int eKindP = gc::B_ZEROGRAD;
            scalar eMixCoef = 0;
            const bool epsWall = (et == "epsilonWallFunction");
            const scalar ecmuP = ed.getOrDefault<scalar>("Cmu", 0.09);
            const scalar ekapP = ed.getOrDefault<scalar>("kappa", 0.41);
            if (epsWall)
            {
                if (ed.getOrDefault<bool>("lowReCorrection", false))
                    FatalErrorInFunction << where << "/epsilon/" << pName
                        << ": epsilonWallFunction lowReCorrection is not supported" << exit(FatalError);
                const word bl = ed.getOrDefault<word>("blending", "stepwise");
                if (bl != "stepwise" && bl != "false" && bl != "off" && bl != "no")
                    FatalErrorInFunction << where << "/epsilon/" << pName
                        << ": epsilonWallFunction blending " << bl << " is not supported (stepwise)" << exit(FatalError);
                eKindP = gc::B_ZEROGRAD;
            }
            else if (et == "fixedValue" || et == "uniformFixedValue") eKindP = gc::B_FIXED;
            else if (et == "zeroGradient" || et == "symmetry" || et == "symmetryPlane" || et == "calculated")
                eKindP = gc::B_ZEROGRAD;
            else if (et == "inletOutlet" || et == "uniformInletOutlet") eKindP = gc::B_INLETOUTLET;
            else if (et == "turbulentMixingLengthDissipationRateInlet")
            {
                if (ed.getOrDefault<word>("k", "k") != "k" || ed.getOrDefault<word>("phi", "phi") != "phi")
                    FatalErrorInFunction << where << "/epsilon/" << pName << ": non-default k/phi is not supported"
                        << exit(FatalError);
                eKindP = gc::B_TURB_MIXINGLEN;
                const scalar L = ed.get<scalar>("mixingLength");
                const scalar cmu = ed.getOrDefault<scalar>("Cmu", fp.turb.Cmu);
                eMixCoef = std::pow(cmu, 0.75)/L;
            }
            else
                FatalErrorInFunction << where << "/epsilon/" << pName << ": epsilon boundary type " << et
                    << " is not supported" << exit(FatalError);

            int nKindP = gc::NUT_CALC;
            const bool nutWall = (nt == "nutkWallFunction");
            if (epsWall != nutWall)
                FatalErrorInFunction << where << "/" << pName
                    << ": epsilonWallFunction and nutkWallFunction must be on the same patches" << exit(FatalError);
            const scalar ncmuP = nd.getOrDefault<scalar>("Cmu", 0.09);
            const scalar nkapP = nd.getOrDefault<scalar>("kappa", 0.41);
            const scalar nEP = nd.getOrDefault<scalar>("E", 9.8);
            if (nutWall) nKindP = gc::NUT_WALL;
            else if (nt == "calculated") nKindP = gc::NUT_CALC;
            else if (nt == "zeroGradient" || nt == "symmetry" || nt == "symmetryPlane") nKindP = gc::NUT_ZEROGRAD;
            else
                FatalErrorInFunction << where << "/nut/" << pName << ": nut boundary type " << nt << " is not supported"
                    << exit(FatalError);

            int aKindP = gc::AT_CALC;
            const scalar aprtP = ad.getOrDefault<scalar>("Prt", 0.85);
            const scalar akapP = ad.getOrDefault<scalar>("kappa", 0.41);
            const scalar aEP = ad.getOrDefault<scalar>("E", 9.8);
            if (at == "compressible::alphatJayatillekeWallFunction" || at == "alphatJayatillekeWallFunction")
            {
                if (!nutWall)
                    FatalErrorInFunction << where << "/alphat/" << pName << ": " << at
                        << " requires epsilonWallFunction/nutkWallFunction on the same patch" << exit(FatalError);
                aKindP = gc::AT_JAYATILLEKE;
            }
            else if (at == "compressible::alphatWallFunction" || at == "alphatWallFunction")
            {
                if (!nutWall)
                    FatalErrorInFunction << where << "/alphat/" << pName << ": " << at
                        << " requires epsilonWallFunction/nutkWallFunction on the same patch" << exit(FatalError);
                aKindP = gc::AT_WALL;
            }
            else if (at == "calculated") aKindP = gc::AT_CALC;
            else if (at == "zeroGradient" || at == "symmetry" || at == "symmetryPlane") aKindP = gc::AT_ZEROGRAD;
            else
                FatalErrorInFunction << where << "/alphat/" << pName << ": alphat boundary type " << at
                    << " is not supported" << exit(FatalError);

            forAll(kp, i)
            {
                const label sl = s0 + i;
                kK[sl] = kKindP;
                kb0[sl] = kp[i];
                if (kKindP == gc::B_FIXED) kV[sl] = kp[i];
                else if (kKindP == gc::B_INLETOUTLET) kV[sl] = refCast<const mixedFvPatchScalarField>(kp).refValue()[i];
                else if (kKindP == gc::B_TURB_INTENSITY) kV[sl] = kIntensity;

                eK[sl] = eKindP;
                epsb0[sl] = ep[i];
                if (eKindP == gc::B_FIXED) eV[sl] = ep[i];
                else if (eKindP == gc::B_INLETOUTLET) eV[sl] = refCast<const mixedFvPatchScalarField>(ep).refValue()[i];
                else if (eKindP == gc::B_TURB_MIXINGLEN) eV[sl] = eMixCoef;

                nK[sl] = nKindP;
                aK[sl] = aKindP;
                eCmu[sl] = ecmuP;
                eKappa[sl] = ekapP;
                nCmu[sl] = ncmuP;
                nKappa[sl] = nkapP;
                nE[sl] = nEP;
                atPrt[sl] = aprtP;
                atKappa[sl] = akapP;
                atE[sl] = aEP;
                sY[sl] = 1.0/dc[i];
                alphatB0[sl] = ap[i];
            }
        }
        E_.fputTurb(f.id, kc, ec, nc_v, ac, kK, kV, eK, eV, nK, aK, eCmu, eKappa, nCmu, nKappa, nE,
                    atPrt, atKappa, atE, sY, alphatB0);
        E_.fput(f.id, gc::F::kb, kb0);
        E_.fput(f.id, gc::F::epsb, epsb0);
    }

    E_.fluidStart(f.id, phiGiven);
    Info<< "    " << where << ": fluid region, " << nc << " cells, " << nS << " boundary slots ("
        << nS - f.nBr << " on region interfaces), div(phi,U) " << fi.conv << ", grad(U) limiter " << fi.limitKU
        << ", nCorrectors " << f.nCorr << ", momentumPredictor " << f.momentumPredictor << endl;
}

void FluidCoupling::updateBCs(FluidRegion& f)
{
    const scalar t = runTime_.timeOutputValue();
    forAll(f.uFlowF1, i)
    {
        const FlowPatchF1& ff = f.uFlowF1[i];
        E_.fputFlowRate(f.id, ff.fpIndex, ff.hvacControlled ? ff.hvacFlowRate : ff.q->value(t));
    }
    forAll(f.uValF1, i)
    {
        const VecPatchF1& vf = f.uValF1[i];
        const vector v = vf.f1->value(t);
        std::vector<double> u(3*vf.n);
        for (label k = 0; k < vf.n; ++k) { u[3*k] = v.x(); u[3*k + 1] = v.y(); u[3*k + 2] = v.z(); }
        E_.fput(f.id, gc::F::uVal, u, 3*vf.s0);
    }
    forAll(f.pValF1, i)
    {
        const ScaPatchF1& pf = f.pValF1[i];
        E_.fput(f.id, gc::F::pVal, std::vector<double>(pf.n, pf.f1->value(t)), pf.s0);
    }
    forAll(f.yValF1, i)
    {
        const ScaPatchF1& yf = f.yValF1[i];
        E_.fputSpecieVal(f.id, yf.specie, std::vector<double>(yf.n, yf.f1->value(t)), yf.s0);
    }
}

static std::string resolveHvacPath(const Time& runTime, const dictionary& d, const word& key)
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

struct PyBridge
{
    int (*IsInitialized)() = nullptr;
    void (*Initialize)() = nullptr;
    int (*RunSimpleString)(const char*) = nullptr;

    void init()
    {
        if (RunSimpleString) return;
        const char* libs[] = {
            "libpython3.12.so.1.0", "libpython3.11.so.1.0", "libpython3.10.so.1.0", "libpython3.so", nullptr
        };
        void* h = nullptr;
        for (int i = 0; libs[i] && !h; ++i)
            h = dlopen(libs[i], RTLD_LAZY | RTLD_GLOBAL);
        if (!h)
            FatalErrorInFunction << "hvac: dlopen libpython3 failed: " << dlerror() << exit(FatalError);
        IsInitialized = reinterpret_cast<int(*)()>(dlsym(h, "Py_IsInitialized"));
        Initialize = reinterpret_cast<void(*)()>(dlsym(h, "Py_Initialize"));
        RunSimpleString = reinterpret_cast<int(*)(const char*)>(dlsym(h, "PyRun_SimpleString"));
        if (!IsInitialized || !Initialize || !RunSimpleString)
            FatalErrorInFunction << "hvac: dlsym Py_* failed" << exit(FatalError);
        if (!IsInitialized()) Initialize();
    }

    void run(const std::string& code)
    {
        fenv_t env;
        fegetenv(&env);
        fedisableexcept(FE_ALL_EXCEPT);
        init();
        const int rc = RunSimpleString(code.c_str());
        feclearexcept(FE_ALL_EXCEPT);
        fesetenv(&env);
        if (rc != 0)
            FatalErrorInFunction << "hvac: Python execution failed" << exit(FatalError);
    }
};
static PyBridge gPy;

void FluidCoupling::setupHvac()
{
    IOobject io("gpuChtProperties", runTime_.constant(), runTime_, IOobject::MUST_READ, IOobject::NO_WRITE,
                IOobject::NO_REGISTER);
    if (!io.typeHeaderOk<IOdictionary>(false) && !isFile(io.objectPath())) return;
    IOdictionary d(io);
    if (!d.isDict("hvac")) return;
    const dictionary& hd = d.subDict("hvac");
    if (!hd.getOrDefault<bool>("enabled", true)) return;

    const std::string caseRoot = std::string(runTime_.path());
    const std::string adapterPath = resolveHvacPath(runTime_, hd, "adapter");
    const std::string configPath = resolveHvacPath(runTime_, hd, "config");
    const std::string mappingPath = resolveHvacPath(runTime_, hd, "mapping");
    const std::string scenario = hd.getOrDefault<word>("scenario", "controlled_design");
    const std::string sensor = hd.getOrDefault<word>("sensor", "occupiedZone");
    const word rname = hd.getOrDefault<word>("region", "air");

    hvacFluidIdx_ = -1;
    forAll(F_, i)
        if (R_[F_[i].r].name == rname) { hvacFluidIdx_ = i; break; }
    if (hvacFluidIdx_ < 0)
        FatalErrorInFunction << "hvac: fluid region " << rname << " not found" << exit(FatalError);

    const FluidRegion& f = F_[hvacFluidIdx_];
    const Region& R = R_[f.r];
    const fvMesh& mesh = *R.mesh;
    const FluidProps& fp = energy_.fluidProps(f.r);
    forAll(fp.species, m)
    {
        if (fp.species[m].name == "waterVapour") hvacVapIdx_ = m;
        else if (fp.species[m].name == "dryAir") hvacDryIdx_ = m;
    }

    const char* pyCode = R"PY(
import ctypes, importlib.util, json, os, sys
import numpy as np

def _gc_str(ptr):
    return ctypes.c_char_p(ptr).value.decode("utf-8")

def _gc_load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

class _GcHvac:
    def __init__(self, p_case, p_adapter, p_config, p_mapping, p_scenario, p_sensor, p_Ab, n_Ab):
        sys.dont_write_bytecode = True
        self.case_root = _gc_str(p_case)
        self.A = _gc_load(_gc_str(p_adapter), "gpucht_native_cht_adapter")
        self.H = self.A.H
        with open(_gc_str(p_config), encoding="utf-8") as fp:
            self.config = json.load(fp)
        with open(_gc_str(p_mapping), encoding="utf-8") as fp:
            self.mapping = json.load(fp)
        self.scenario = _gc_str(p_scenario)
        self.sensor = _gc_str(p_sensor)
        self.slices = {}
        self.A_b = np.ctypeslib.as_array((ctypes.c_double * n_Ab).from_address(p_Ab)).copy() if n_Ab > 0 else np.zeros(0)
        self.room_cells = {}
        self.room_of = {u["fcu_id"]: u["room_id"] for u in self.config["units"]}
        self.state = None
        self.history = []

    def add_patch(self, p_name, pi, s0, s1):
        self.slices[_gc_str(p_name)] = (pi, slice(s0, s1))

    def add_room(self, p_rid, p_cells, p_vols, n):
        c = np.ctypeslib.as_array((ctypes.c_int32 * n).from_address(p_cells)).copy() if n > 0 else np.zeros(0, dtype=np.int32)
        v = np.ctypeslib.as_array((ctypes.c_double * n).from_address(p_vols)).copy() if n > 0 else np.zeros(0, dtype=np.float64)
        self.room_cells[_gc_str(p_rid)] = (c, v)

    def finalize(self):
        need = ({self.room_of[u["fcu_id"]] for u in self.mapping["fcu_ports"]}
                if self.mapping.get("component_only", False)
                else {u["room_id"] for u in self.config["units"]})
        missing = sorted(need - set(self.room_cells))
        if missing and self.sensor == "occupiedZone":
            raise RuntimeError("hvac: no room cellZone for rooms " + ", ".join(missing))

    def step(self, t0, dt, has_weather, Ta, w_out, p_out, p_phi, p_Tb, p_Yv, p_pb, n_br, p_Tc, n_c,
             p_kind, p_mdot, p_tk, p_yv, n_p, p_msg):
        phi_b = np.ctypeslib.as_array((ctypes.c_double * n_br).from_address(p_phi))
        T_b = np.ctypeslib.as_array((ctypes.c_double * n_br).from_address(p_Tb))
        Yv = np.ctypeslib.as_array((ctypes.c_double * n_br).from_address(p_Yv)) if p_Yv != 0 else None
        p_b = np.ctypeslib.as_array((ctypes.c_double * n_br).from_address(p_pb))
        T_c = np.ctypeslib.as_array((ctypes.c_double * n_c).from_address(p_Tc))
        A = self.A_b
        H = self.H
        returns = {}
        for u in self.mapping["fcu_ports"]:
            dry = vap = hflux = pdry = area = aT = aw = ap = 0.0
            for e in u["return"]:
                _, sl = self.slices[e["patch"]]
                ph = np.maximum(phi_b[sl], 0.0)
                Y = Yv[sl] if Yv is not None else np.zeros(sl.stop - sl.start)
                d_ = ph * (1 - Y)
                w_ = Y / (1 - Y)
                Tc = T_b[sl] - 273.15
                dry += d_.sum()
                vap += (d_ * w_).sum()
                hflux += (d_ * np.array([H.enthalpy(t, w) for t, w in zip(Tc, w_)])).sum() if d_.sum() > 0 else 0.0
                pdry += (d_ * p_b[sl]).sum()
                a = A[sl]
                area += a.sum()
                aT += (a * Tc).sum()
                aw += (a * w_).sum()
                ap += (a * p_b[sl]).sum()
            if dry > 1e-12:
                w = vap / dry
                r = dict(T_C=H.temperature_from_h_w(hflux / dry, w), w=w, p_Pa=pdry / dry,
                         observed_return_dry_air_mass_kg_s=dry, weighting="outgoing dry-air-mass weighted (solved CFD)")
            else:
                r = dict(T_C=aT / area, w=aw / area, p_Pa=ap / area, observed_return_dry_air_mass_kg_s=0.0,
                         weighting="zero-flow area mean of the return face (system off or start-up)")
            if self.sensor == "occupiedZone":
                cells, V = self.room_cells[self.room_of[u["fcu_id"]]]
                r["sensor_C"] = float((T_c[cells] * V).sum() / V.sum()) - 273.15
            returns[u["fcu_id"]] = r
        for fid, st in self.mapping.get("component_reservoir_returns", {}).items():
            if not self.mapping.get("component_only", False):
                raise RuntimeError("hvac: reservoir returns are only allowed with component_only true")
            returns[fid] = dict(st, weighting="fixed component-test reservoir, not solved CFD")
        outdoor = dict(T_C=Ta - 273.15, w=w_out, p_Pa=p_out) if has_weather else None
        if self.state is not None and abs(self.state["next_physical_time_s"] - t0) > 1e-6:
            self.state = dict(self.state, next_physical_time_s=t0)
        res = self.A.coupled_step(self.config, self.mapping, returns, t0, dt, state=self.state,
                                  scenario=self.scenario, outdoor=outdoor)
        self.state = dict(next_physical_time_s=res["next_physical_time_s"], controllers=res["controllers"])
        relief = {u["fcu_id"]: u.get("relief_dry_air_mass_kg_s", 0.0) for u in self.mapping["fcu_ports"]}
        oa_on = res["fau"]["dry_air_mass_kg_s"] > 0
        ret_unit = {}
        for u in self.mapping["fcu_ports"]:
            for e in u["return"]:
                ret_unit[e["patch"]] = (u["fcu_id"], e["fraction"])
        out_kind = np.ctypeslib.as_array((ctypes.c_int32 * n_p).from_address(p_kind))
        out_mdot = np.ctypeslib.as_array((ctypes.c_double * n_p).from_address(p_mdot))
        out_tk = np.ctypeslib.as_array((ctypes.c_double * n_p).from_address(p_tk))
        out_yv = np.ctypeslib.as_array((ctypes.c_double * n_p).from_address(p_yv))
        for p in res["boundary_ports"]:
            pn = p["patch"]
            if pn not in self.slices:
                raise RuntimeError(f"hvac: patch {pn} not found")
            pi, _ = self.slices[pn]
            m = float(p["moist_mass_kg_s"])
            is_supply = (p["kind"] == "supply")
            if p["kind"] == "return":
                fid, frac = ret_unit[pn]
                if oa_on:
                    m = m + relief[fid] * frac * (1.0 + returns[fid]["w"])
            out_kind[pi] = 2 if is_supply else 1
            out_mdot[pi] = m
            if is_supply:
                out_tk[pi] = float(p["T_C"] + 273.15)
                out_yv[pi] = float(p["Yv"])
        sr = res["frozen_runtime_result"]
        alloc = sr["allocation"]
        tot_m = sum(p["moist_mass_kg_s"] for p in res["boundary_ports"] if p["kind"] == "supply")
        row = dict(t=t0, enabled=bool(sr["enabled"]), supply_moist_kg_s=tot_m,
                   FCU_water_W=alloc["FCU_used_water_cooling_W"], plant_W=alloc["total_served_cooling_W"],
                   capacity_scale=alloc["capacity_scale"], FAU_T_C=res["fau"]["T_C"],
                   sensors=res["room_sensor_C"])
        self.history.append(row)
        d = os.path.join(self.case_root, "postProcessing", "hvac")
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, "hvac.json.tmp")
        with open(tmp, "w") as fp:
            json.dump(self.history, fp, indent=0)
        os.replace(tmp, os.path.join(d, "hvac.json"))
        msg = ("hvac: t = %g s enabled %s, supply %.2f kg/s, FCU water %.1f kW, plant %.1f kW, sensors %.2f..%.2f C"
               % (t0, row["enabled"], tot_m, row["FCU_water_W"] / 1e3, row["plant_W"] / 1e3,
                  min(row["sensors"].values()), max(row["sensors"].values())))
        mb = msg.encode("utf-8")[:1020]
        ctypes.memmove(p_msg, mb, len(mb))
        ctypes.c_char.from_address(p_msg + len(mb)).value = b"\0"
)PY";
    gPy.run(pyCode);

    std::vector<double> A_b(f.nBr, 0.0);
    forAll(mesh.boundary(), pi)
    {
        if (R.patches[pi].kind != PatchMap::BOUNDARY) continue;
        const label s0 = f.slot0[pi];
        const scalarField& magSf = mesh.boundary()[pi].magSf();
        forAll(magSf, i) A_b[s0 + i] = magSf[i];
    }

    auto uptr = [](const void* p) -> unsigned long long
    {
        return static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(p));
    };

    char cmd[512];
    std::snprintf
    (
        cmd, sizeof(cmd),
        "_gc_hvac = _GcHvac(%llu, %llu, %llu, %llu, %llu, %llu, %llu, %d)",
        uptr(caseRoot.c_str()),
        uptr(adapterPath.c_str()),
        uptr(configPath.c_str()),
        uptr(mappingPath.c_str()),
        uptr(scenario.c_str()),
        uptr(sensor.c_str()),
        uptr(A_b.data()),
        int(A_b.size())
    );
    gPy.run(cmd);

    forAll(mesh.boundary(), pi)
    {
        if (R.patches[pi].kind != PatchMap::BOUNDARY) continue;
        const label s0 = f.slot0[pi];
        const label n = mesh.boundary()[pi].size();
        const std::string pname = mesh.boundary()[pi].name();
        std::snprintf
        (
            cmd, sizeof(cmd),
            "_gc_hvac.add_patch(%llu, %d, %d, %d)",
            uptr(pname.c_str()),
            int(pi), int(s0), int(s0 + n)
        );
        gPy.run(cmd);
    }

    const cellZoneMesh& cz = mesh.cellZones();
    forAll(cz, k)
    {
        const word& zn = cz[k].name();
        if (zn.size() <= 5 || zn.substr(0, 5) != "room_") continue;
        const labelList& cells = cz[k];
        if (cells.empty()) continue;
        std::vector<int32_t> zc(cells.begin(), cells.end());
        std::sort(zc.begin(), zc.end());
        scalar z0 = GREAT;
        for (int32_t c : zc) z0 = min(z0, mesh.C()[c].z());
        std::vector<int32_t> occ;
        std::vector<double> occV;
        for (int32_t c : zc)
        {
            const scalar z = mesh.C()[c].z();
            if (z >= z0 + 0.1 && z <= z0 + 1.7)
            {
                occ.push_back(c);
                occV.push_back(mesh.V()[c]);
            }
        }
        const std::string rid = zn.substr(5);
        std::snprintf
        (
            cmd, sizeof(cmd),
            "_gc_hvac.add_room(%llu, %llu, %llu, %d)",
            uptr(rid.c_str()),
            uptr(occ.data()),
            uptr(occV.data()),
            int(occ.size())
        );
        gPy.run(cmd);
    }
    gPy.run("_gc_hvac.finalize()");
    hasHvac_ = true;
}

void FluidCoupling::hvacControl(scalar t0, scalar dt)
{
    if (!hasHvac_) return;
    FluidRegion& f = F_[hvacFluidIdx_];
    const Region& R = R_[f.r];
    fvMesh& mesh = *R.mesh;
    const label nBr = f.nBr;
    std::vector<double> phiB = E_.fget(f.id, gc::F::phiB, 0, nBr);
    std::vector<double> Tb_r = E_.get(gc::A::TbB, R.b0, nBr);
    std::vector<double> Yc_v, Yb_v;
    if (hvacVapIdx_ >= 0) E_.fgetSpecie(f.id, hvacVapIdx_, Yc_v, Yb_v);
    const std::vector<double> pAbs = E_.fget(f.id, gc::F::pAbs);
    std::vector<double> p_b(nBr, 0.0);
    forAll(mesh.boundary(), pi)
    {
        if (R.patches[pi].kind != PatchMap::BOUNDARY) continue;
        const label s0 = f.slot0[pi];
        const labelUList& fc = mesh.boundary()[pi].faceCells();
        forAll(fc, i) p_b[s0 + i] = pAbs[fc[i]];
    }
    std::vector<double> Tc = E_.get(gc::A::T, R.c0, mesh.nCells());

    const bool hasW = energy_.hasBuilding();
    gc::WeatherState W;
    if (hasW) W = E_.buildingWeather(t0);

    const label nP = mesh.boundary().size();
    std::vector<int32_t> outKind(nP, 0);
    std::vector<double> outMdot(nP, 0.0), outTk(nP, 0.0), outYv(nP, 0.0);
    char outMsg[1024] = {0};

    auto uptr = [](const void* p) -> unsigned long long
    {
        return static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(p));
    };

    char cmd[768];
    std::snprintf
    (
        cmd, sizeof(cmd),
        "_gc_hvac.step(%.17g, %.17g, %d, %.17g, %.17g, %.17g, %llu, %llu, %llu, %llu, %d, %llu, %d, %llu, %llu, %llu, %llu, %d, %llu)",
        double(t0), double(dt), hasW ? 1 : 0, W.Ta, W.w, W.p,
        uptr(phiB.data()),
        uptr(Tb_r.data()),
        hvacVapIdx_ >= 0 ? uptr(Yb_v.data()) : 0ULL,
        uptr(p_b.data()),
        int(nBr),
        uptr(Tc.data()),
        int(Tc.size()),
        uptr(outKind.data()),
        uptr(outMdot.data()),
        uptr(outTk.data()),
        uptr(outYv.data()),
        int(nP),
        uptr(outMsg)
    );
    gPy.run(cmd);

    forAll(mesh.boundary(), pi)
    {
        if (outKind[pi] == 0) continue;
        forAll(f.uFlowF1, k)
        {
            if (f.uFlowF1[k].patch == pi)
            {
                f.uFlowF1[k].hvacControlled = true;
                f.uFlowF1[k].hvacFlowRate = outMdot[pi];
                break;
            }
        }
        if (outKind[pi] == 2)
        {
            const label n = mesh.boundary()[pi].size();
            const label b0 = R.patches[pi].b0;
            const label s0 = f.slot0[pi];
            const double Tk = outTk[pi], yv = outYv[pi];
            energy_.T(f.r).boundaryFieldRef()[pi] == Tk;
            const std::vector<double> vTk(n, Tk);
            E_.put(gc::A::bVal, vTk, b0);
            E_.put(gc::A::TbB, vTk, b0);
            if (hvacVapIdx_ >= 0) E_.fputSpecieVal(f.id, hvacVapIdx_, std::vector<double>(n, yv), s0);
            if (hvacDryIdx_ >= 0) E_.fputSpecieVal(f.id, hvacDryIdx_, std::vector<double>(n, 1.0 - yv), s0);
        }
    }
    updateBCs(f);
    E_.fluidUpdateBoundary(f.id, true);
    Info<< outMsg << endl;
}

scalar FluidCoupling::courantNo(scalar dt, scalar& meanCo) const
{
    scalar CoNum = -GREAT;
    meanCo = 0.0;
    forAll(F_, i)
    {
        const FluidRegion& f = F_[i];
        const fvMesh& mesh = *R_[f.r].mesh;
        double meanCoR = 0.0;
        const double coR = E_.fluidCourantNo(f.id, dt, meanCoR);
        Info<< "Region: " << mesh.name() << " Courant Number mean: " << meanCoR
            << " max: " << coR << endl;
        CoNum = max(CoNum, scalar(coR));
        meanCo = max(meanCo, scalar(meanCoR));
    }
    return CoNum;
}

// ----------------------------------------------------------------------------------------- time step
void FluidCoupling::rhoToEnergy()
{
    forAll(F_, i) E_.fluidRhoToEnergy(F_[i].id);
}

void FluidCoupling::storeOld(scalar dt)
{
    forAll(F_, i) E_.fluidStoreOld(F_[i].id, dt, F_[i].frozenFlow, nOuterCorr_ != 1);
}

void FluidCoupling::momentum(scalar dt, bool finalIter)
{
    forAll(F_, i)
    {
        FluidRegion& f = F_[i];
        const fvMesh& mesh = *R_[f.r].mesh;
        Info<< "\nSolving for fluid region " << mesh.name() << endl;
        if (f.frozenFlow) continue;
        updateBCs(f);
        const scalar alpha = relaxFactor(mesh, "U", finalIter, false);
        const gc::SolverCtl ctl = readSolver(mesh, finalIter ? "UFinal" : "U", false);
        gc::SolveInfo info[3];
        E_.fluidMomentum(f.id, dt, alpha, ctl, f.momentumPredictor, info);
        double uRes = 0.0;
        const Vector<label> sd = mesh.solutionD();
        for (int k = 0; k < 3; ++k)
            if (sd[k] > 0) uRes = std::max(uRes, info[k].res0);
        f.lastRes["U"] = uRes;
        if (f.momentumPredictor)
        {
            const char* c = "xyz";
            for (int k = 0; k < 3; ++k)
                if (sd[k] > 0) logSolve(ctl.name, word(std::string("U") + c[k]), info[k]);
        }
    }
}

void FluidCoupling::species(scalar dt, bool finalIter)
{
    forAll(F_, i)
    {
        FluidRegion& f = F_[i];
        if (f.frozenFlow || f.Y.empty()) continue;
        const fvMesh& mesh = *R_[f.r].mesh;
        const FluidProps& fp = energy_.fluidProps(f.r);
        energy_.updateSpecieSources(f.r, f.id);
        std::vector<double> alpha(fp.species.size(), 0.0);
        forAll(fp.species, m)
            alpha[m] = relaxFactor(mesh, fp.species[m].name, finalIter, false);
        const gc::SolverCtl ctl = readSolver(mesh, "Yi", false);
        std::vector<gc::SolveInfo> info;
        E_.fluidSpecies(f.id, dt, alpha, ctl, info);
        forAll(fp.species, m)
        {
            if (m == fp.inertIndex) continue;
            f.lastRes[fp.species[m].name] = info[m].res0;
            logSolve(ctl.name, fp.species[m].name, info[m]);
        }
    }
}

void FluidCoupling::toEnergy(scalar dt)
{
    forAll(F_, i) E_.fluidToEnergy(F_[i].id, dt, energy_.fluidProps(F_[i].r).dpdt);
}

void FluidCoupling::pressure(scalar dt, bool finalIter)
{
    forAll(F_, i)
    {
        FluidRegion& f = F_[i];
        if (f.frozenFlow) continue;
        const fvMesh& mesh = *R_[f.r].mesh;
        const scalar pRelax = relaxFactor(mesh, "p_rgh", finalIter, true);
        for (label corr = 0; corr < f.nCorr; ++corr)
        {
            const bool fin = finalIter && corr == f.nCorr - 1;
            const gc::SolverCtl ctl = readSolver(mesh, fin ? "p_rghFinal" : "p_rgh", true);
            double cont[2] = {0, 0};
            const gc::SolveInfo si = E_.fluidPressure(f.id, dt, ctl, pRelax, f.nNonOrth, cont);
            if (corr == 0) f.lastRes["p_rgh"] = si.res0;
            logSolve(ctl.name, "p_rgh", si);
            f.cumulativeContErr += cont[1];
            Info<< "time step continuity errors (" << mesh.name() << ") : sum local = " << cont[0]
                << ", global = " << cont[1] << ", cumulative = " << f.cumulativeContErr << endl;
        }
        // solveFluid.H: rho = thermo.rho()
        E_.fluidUpdateRho(f.id);
        const std::vector<double> r = E_.fget(f.id, gc::F::rho);
        double mn = 1e300, mx = -1e300;
        for (const double v : r) { mn = std::min(mn, v); mx = std::max(mx, v); }
        Info<< "Min/max rho:" << mn << ' ' << mx << endl;
    }
}

void FluidCoupling::turbulence(scalar dt, bool finalIter)
{
    forAll(F_, i)
    {
        FluidRegion& f = F_[i];
        if (f.frozenFlow || !f.turbulent) continue;
        const fvMesh& mesh = *R_[f.r].mesh;
        const scalar alphaK = relaxFactor(mesh, "k", finalIter, false);
        const scalar alphaEps = relaxFactor(mesh, "epsilon", finalIter, false);
        const gc::SolverCtl ctlK = readSolver(mesh, finalIter ? "kFinal" : "k", false, true);
        const gc::SolverCtl ctlEps = readSolver(mesh, finalIter ? "epsilonFinal" : "epsilon", false, true);
        gc::SolveInfo infoEps, infoK;
        E_.fluidTurbulence(f.id, dt, alphaK, alphaEps, ctlK, ctlEps, infoEps, infoK);
        f.lastRes["epsilon"] = infoEps.res0;
        f.lastRes["k"] = infoK.res0;
        logSolve(ctlEps.name, "epsilon", infoEps);
        logSolve(ctlK.name, "k", infoK);
    }
}

bool FluidCoupling::outerConverged(const dictionary& ocrc, scalar hRes) const
{
    for (const entry& e : ocrc)
    {
        if (!e.isDict()) continue;
        const word& fld = e.keyword();
        const dictionary& fd = e.dict();
        const scalar tol = fd.getOrDefault<scalar>("tolerance", 0.0);
        const scalar relTol = fd.getOrDefault<scalar>("relTol", 0.0);
        if (relTol != 0)
            FatalErrorInFunction << "outerCorrectorResidualControl/" << fld
                << ": non-zero relTol is not supported (use relTol 0)" << exit(FatalError);
        if (fld == "h" && hRes > tol) return false;
        forAll(F_, i)
        {
            const auto it = F_[i].lastRes.find(fld);
            if (it != F_[i].lastRes.end() && it->second > tol) return false;
        }
    }
    return true;
}

// ----------------------------------------------------------------------------------------- output
void FluidCoupling::writeBack()
{
    forAll(F_, i)
    {
        FluidRegion& f = F_[i];
        const fvMesh& mesh = *R_[f.r].mesh;
        const std::vector<double> Uc = E_.fget(f.id, gc::F::U), Ub = E_.fget(f.id, gc::F::Ub);
        const std::vector<double> pc = E_.fget(f.id, gc::F::p), pa = E_.fget(f.id, gc::F::pAbs);
        const std::vector<double> paB = E_.fget(f.id, gc::F::pAbsB), pG = E_.fget(f.id, gc::F::pGrad);
        const std::vector<double> pV = E_.fget(f.id, gc::F::pVal);
        const std::vector<double> rc = E_.fget(f.id, gc::F::rho), rB = E_.fget(f.id, gc::F::rhoB);
        const std::vector<double> ph = E_.fget(f.id, gc::F::phi), phB = E_.fget(f.id, gc::F::phiB);
        volVectorField& U = f.U();
        volScalarField& p_rgh = f.p_rgh();
        volScalarField& p = f.p();
        volScalarField& rho = f.rho();
        surfaceScalarField& phi = f.phi();
        forAll(f.uFlowF1, k)
        {
            const FlowPatchF1& ff = f.uFlowF1[k];
            if (!ff.hvacControlled) continue;
            const label pi = ff.patch;
            dictionary ud = patchDict(U.boundaryField()[pi]);
            OStringStream os;
            os << "massFlowRate constant " << ff.hvacFlowRate << ";";
            IStringStream is(os.str());
            ud.set(entry::New(is));
            U.boundaryFieldRef().set
            (
                pi,
                fvPatchVectorField::New(mesh.boundary()[pi], U.internalField(), ud)
            );
        }
        forAll(U, c)
        {
            U.primitiveFieldRef()[c] = vector(Uc[3*c], Uc[3*c + 1], Uc[3*c + 2]);
            p_rgh.primitiveFieldRef()[c] = pc[c];
            p.primitiveFieldRef()[c] = pa[c];
            rho.primitiveFieldRef()[c] = rc[c];
        }
        forAll(phi, j) phi.primitiveFieldRef()[j] = ph[j];
        forAll(mesh.boundary(), pi)
        {
            const label s0 = f.slot0[pi];
            if (s0 < 0) continue;
            const label n = mesh.boundary()[pi].size();
            vectorField ub(n);
            scalarField pr(n), pab(n), rb(n), fb(n);
            forAll(ub, k)
            {
                const label sl = s0 + k;
                ub[k] = vector(Ub[3*sl], Ub[3*sl + 1], Ub[3*sl + 2]);
                pab[k] = paB[sl];
                rb[k] = rB[sl];
                fb[k] = phB[sl];
            }
            fvPatchVectorField& up = U.boundaryFieldRef()[pi];
            if (isA<mixedFvPatchVectorField>(up))
            {
                mixedFvPatchVectorField& mp = refCast<mixedFvPatchVectorField>(up);
                forAll(ub, k) mp.valueFraction()[k] = phB[s0 + k] < 0.0 ? 1.0 : 0.0;
            }
            up == ub;
            p.boundaryFieldRef()[pi] == pab;
            rho.boundaryFieldRef()[pi] == rb;
            phi.boundaryFieldRef()[pi] == fb;
            // p_rgh on the patch: the slot value the device used (fixed value / cell + gradient / cell value)
            fvPatchScalarField& pp = p_rgh.boundaryFieldRef()[pi];
            const labelUList& fc = mesh.boundary()[pi].faceCells();
            const scalarField& dc = mesh.boundary()[pi].deltaCoeffs();
            const word& pt = pp.type();
            if (pt == "fixedFluxPressure")
            {
                scalarField& gr = refCast<fixedGradientFvPatchScalarField>(pp).gradient();
                forAll(pr, k)
                {
                    gr[k] = pG[s0 + k];
                    pr[k] = pc[fc[k]] + gr[k]/dc[k];
                }
                pp == pr;
            }
            else if (pt == "zeroGradient" || pt == "symmetry" || pt == "symmetryPlane")
            {
                forAll(pr, k) pr[k] = pc[fc[k]];
                pp == pr;
            }
            else if (pt == "fixedValue" || pt == "uniformFixedValue" || pt == "prghPressure" || pt == "prghTotalPressure")
            {
                forAll(pr, k) pr[k] = pV[s0 + k];
                pp == pr;
            }
        }
        forAll(f.Y, m)
        {
            std::vector<double> Yc, Yb;
            E_.fgetSpecie(f.id, m, Yc, Yb);
            volScalarField& Ym = f.Y[m];
            forAll(Ym, c) Ym.primitiveFieldRef()[c] = Yc[c];
            forAll(mesh.boundary(), pi)
            {
                const label s0 = f.slot0[pi];
                if (s0 < 0) continue;
                const label n = mesh.boundary()[pi].size();
                scalarField yb(n);
                forAll(yb, k) yb[k] = Yb[s0 + k];
                fvPatchScalarField& yp = Ym.boundaryFieldRef()[pi];
                if (yp.type() == "inletOutlet" || yp.type() == "uniformInletOutlet")
                {
                    mixedFvPatchScalarField& mp = refCast<mixedFvPatchScalarField>(yp);
                    forAll(yb, k) mp.valueFraction()[k] = phB[s0 + k] < 0.0 ? 1.0 : 0.0;
                }
                yp == yb;
            }
        }
        if (f.turbulent)
        {
            const std::vector<double> kc = E_.fget(f.id, gc::F::k), kb = E_.fget(f.id, gc::F::kb);
            const std::vector<double> ec = E_.fget(f.id, gc::F::eps), eb = E_.fget(f.id, gc::F::epsb);
            const std::vector<double> nc_v = E_.fget(f.id, gc::F::nut), nb = E_.fget(f.id, gc::F::nutB);
            const std::vector<double> ac = E_.fget(f.id, gc::F::alphat), ab = E_.fget(f.id, gc::F::alphatB);
            volScalarField& k = f.k();
            volScalarField& eps = f.epsilon();
            volScalarField& nut = f.nut();
            volScalarField& alphat = f.alphat();
            forAll(k, c)
            {
                k.primitiveFieldRef()[c] = kc[c];
                eps.primitiveFieldRef()[c] = ec[c];
                nut.primitiveFieldRef()[c] = nc_v[c];
                alphat.primitiveFieldRef()[c] = ac[c];
            }
            forAll(mesh.boundary(), pi)
            {
                const label s0 = f.slot0[pi];
                if (s0 < 0) continue;
                const label n = mesh.boundary()[pi].size();
                scalarField kbf(n), ebf(n), nbf(n), abf(n);
                forAll(kbf, j)
                {
                    const label sl = s0 + j;
                    kbf[j] = kb[sl];
                    ebf[j] = eb[sl];
                    nbf[j] = nb[sl];
                    abf[j] = ab[sl];
                }
                fvPatchScalarField& kp = k.boundaryFieldRef()[pi];
                fvPatchScalarField& ep = eps.boundaryFieldRef()[pi];
                if (isA<mixedFvPatchScalarField>(kp))
                {
                    mixedFvPatchScalarField& mp = refCast<mixedFvPatchScalarField>(kp);
                    forAll(kbf, j) mp.valueFraction()[j] = phB[s0 + j] < 0.0 ? 1.0 : 0.0;
                }
                if (isA<mixedFvPatchScalarField>(ep))
                {
                    mixedFvPatchScalarField& mp = refCast<mixedFvPatchScalarField>(ep);
                    forAll(ebf, j) mp.valueFraction()[j] = phB[s0 + j] < 0.0 ? 1.0 : 0.0;
                }
                kp == kbf;
                ep == ebf;
                nut.boundaryFieldRef()[pi] == nbf;
                alphat.boundaryFieldRef()[pi] == abf;
            }
        }
    }
}

} // namespace gpuCht
} // namespace Foam
