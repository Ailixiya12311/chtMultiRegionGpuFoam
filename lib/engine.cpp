#include "api.h"
#include "building.h"
#include "energy.h"
#include "flow.h"

#include <memory>

namespace gc
{

struct Engine::Impl
{
    DMesh m;
    Energy E;
    Building bld;
    std::vector<std::unique_ptr<Flow>> fl;

    Flow& flow(int id)
    {
        if (id < 0 || id >= int(fl.size())) fatal("Engine: no fluid %d", id);
        return *fl[id];
    }

    dvec<double>& farr(int id, F a)
    {
        Flow& f = flow(id);
        switch (a)
        {
            case F::U: return f.U;
            case F::p: return f.p;
            case F::pAbs: return f.pAbs;
            case F::rho: return f.rho;
            case F::dpdt: return f.dpdt;
            case F::mu: return f.mu;
            case F::eRho0: return f.eRho0;
            case F::eT0: return f.eT0;
            case F::eBeta: return f.eBeta;
            case F::k: return f.k;
            case F::eps: return f.eps;
            case F::nut: return f.nut;
            case F::alphat: return f.alphat;
            case F::Ub: return f.Ub;
            case F::rhoB: return f.rhoB;
            case F::pAbsB: return f.pAbsB;
            case F::phiB: return f.phiB;
            case F::muB: return f.muB;
            case F::eRho0B: return f.eRho0B;
            case F::eT0B: return f.eT0B;
            case F::eBetaB: return f.eBetaB;
            case F::kb: return f.kb;
            case F::epsb: return f.epsb;
            case F::nutB: return f.nutB;
            case F::alphatB: return f.alphatB;
            case F::uVal: return f.uVal;
            case F::uVf: return f.uVf;
            case F::pVal: return f.pVal;
            case F::pGrad: return f.pGrad;
            case F::p0Val: return f.p0Val;
            case F::phi: return f.phi;
        }
        fatal("Engine: unknown fluid array");
    }

    dvec<double>& arr(A a)
    {
        switch (a)
        {
            case A::T: return E.T;
            case A::rho: return E.rho;
            case A::cp: return E.cp;
            case A::e: return E.e;
            case A::kappa: return E.kappa;
            case A::alphat: return E.alphat;
            case A::Su: return E.Su;
            case A::Sp: return E.Sp;
            case A::rLimitK: return E.rLimitK;
            case A::rPsi: return E.rPsi;
            case A::rRelax: return E.rRelax;
            case A::bVal: return E.bVal;
            case A::bGrad: return E.bGrad;
            case A::bVf: return E.bVf;
            case A::cpB: return E.cpB;
            case A::eB: return E.eB;
            case A::kappaB: return E.kappaB;
            case A::alphatB: return E.alphatB;
            case A::TbB: return E.TbB;
            case A::Rlayer: return E.Rlayer;
            case A::qIface: return E.qIface;
            case A::kappaWall: return E.kappaWall;
            case A::TA: return E.TA;
            case A::TB: return E.TB;
            case A::phi: return E.phi;
        }
        fatal("Engine: unknown array");
    }
};

Engine::Engine(const MeshIn& mi) : p_(new Impl)
{
    p_->m.build(mi);
    p_->E.init(p_->m, int(mi.regions.size()));
}

Engine::~Engine() = default;

int Engine::nC() const { return p_->m.nC; }
int Engine::nI() const { return p_->m.nI; }
int Engine::nB() const { return p_->m.nB; }

void Engine::put(A a, const std::vector<double>& v, long offset)
{
    dvec<double>& d = p_->arr(a);
    if (offset < 0 || offset + long(v.size()) > long(d.size()))
        fatal("Engine::put: %ld values at offset %ld exceed the array size %ld", long(v.size()), offset, long(d.size()));
    thrust::copy(v.begin(), v.end(), d.begin() + offset);
}

std::vector<double> Engine::get(A a, long offset, long n) const
{
    const dvec<double>& d = p_->arr(a);
    if (n < 0) n = long(d.size()) - offset;
    if (offset < 0 || offset + n > long(d.size()))
        fatal("Engine::get: range %ld+%ld exceeds the array size %ld", offset, n, long(d.size()));
    std::vector<double> h(n);
    thrust::copy(d.begin() + offset, d.begin() + offset + n, h.begin());
    return h;
}

void Engine::minMax(A a, long offset, long n, double& mn, double& mx) const
{
    const double* d = ptr(p_->arr(a)) + offset;
    mn = min_n(n, [=] HD (long i) { return d[i]; });
    mx = max_n(n, [=] HD (long i) { return d[i]; });
}

void Engine::putFluid(const std::vector<char>& fluid)
{
    if (long(fluid.size()) != p_->m.nC) fatal("Engine::putFluid: size %ld != nC", long(fluid.size()));
    thrust::copy(fluid.begin(), fluid.end(), p_->E.fluid.begin());
}

void Engine::putBKind(const std::vector<int>& kind, long offset)
{
    if (offset + long(kind.size()) > p_->m.nB) fatal("Engine::putBKind: range exceeds nB");
    thrust::copy(kind.begin(), kind.end(), p_->E.bKind.begin() + offset);
}

void Engine::energyInit()
{
    p_->E.boundaryState();
}

void Engine::energyStoreOld()
{
    p_->E.storeOld();
}

SolveInfo Engine::energySolve(double dt, const SolverCtl& ctl, bool relaxOn)
{
    return p_->E.solve(dt, ctl, relaxOn);
}

void Engine::setFixedT(const std::vector<int>& cells, const std::vector<double>& T)
{
    p_->E.fixCells = to_dev(cells);
    p_->E.fixVals = to_dev(T);
}

std::vector<double> Engine::boundaryT() const
{
    p_->E.boundaryState();
    return to_host(p_->E.TbB);
}

void Engine::interfaceState(std::vector<double>& TA, std::vector<double>& TB, std::vector<double>& gA,
                            std::vector<double>& gB) const
{
    p_->E.boundaryState();
    TA = to_host(p_->E.TA);
    TB = to_host(p_->E.TB);
    gA = to_host(p_->E.gA);
    gB = to_host(p_->E.gB);
}

double Engine::memGB() const
{
    return mem_gb();
}

void Engine::energyBoundaryState()
{
    p_->E.boundaryState();
}

void Engine::setEnergyConvection(int conv, double k, bool bounded)
{
    p_->E.conv = conv;
    p_->E.convK = k;
    p_->E.bounded = bounded;
}

int Engine::addFluid(const FluidIn& fi)
{
    std::unique_ptr<Flow> f(new Flow());
    f->init(p_->m, p_->E, fi.region, fi.ifFaces, fi.ifSideA);
    for (int k = 0; k < 3; ++k) { f->g[k] = fi.g[k]; f->solveDir[k] = fi.solveDir[k]; }
    f->ghRef = fi.ghRef;
    f->conv = fi.conv;
    f->bounded = fi.bounded;
    f->limitKU = fi.limitKU;
    f->psiU = fi.psiU;
    f->psiP = fi.psiP;
    f->psiSnP = fi.psiSnP;
    f->psiSnRho = fi.psiSnRho;
    f->pRefCell = fi.pRefCell;
    f->pRefValue = fi.pRefValue;
    f->initSpecies(fi);
    f->initTurbulence(fi);
    p_->fl.push_back(std::move(f));
    return int(p_->fl.size()) - 1;
}

int Engine::nSlots(int id) const { return p_->flow(id).nS; }

void Engine::fput(int id, F a, const std::vector<double>& v, long offset)
{
    dvec<double>& d = p_->farr(id, a);
    if (offset < 0 || offset + long(v.size()) > long(d.size()))
        fatal("Engine::fput: %ld values at offset %ld exceed the array size %ld", long(v.size()), offset, long(d.size()));
    thrust::copy(v.begin(), v.end(), d.begin() + offset);
}

std::vector<double> Engine::fget(int id, F a, long offset, long n) const
{
    const dvec<double>& d = p_->farr(id, a);
    if (n < 0) n = long(d.size()) - offset;
    if (offset < 0 || offset + n > long(d.size())) fatal("Engine::fget: range exceeds the array size");
    std::vector<double> h(n);
    thrust::copy(d.begin() + offset, d.begin() + offset + n, h.begin());
    return h;
}

void Engine::fputKinds(int id, const std::vector<int>& uKind, const std::vector<int>& pKind,
                       const std::vector<char>& uAssign, const std::vector<char>& uFixes)
{
    Flow& f = p_->flow(id);
    if (long(uKind.size()) != f.nS || long(pKind.size()) != f.nS || long(uAssign.size()) != f.nS
     || long(uFixes.size()) != f.nS)
        fatal("Engine::fputKinds: sizes do not match the %d slots", f.nS);
    f.uKind = to_dev(uKind);
    f.pKind = to_dev(pKind);
    f.uAssign = to_dev(uAssign);
    f.uFixes = to_dev(uFixes);
}

void Engine::fputFlowPatches(int id, const std::vector<FlowPatchIn>& fp)
{
    Flow& f = p_->flow(id);
    f.nFP = int(fp.size());
    std::vector<int> s0(f.nFP), n(f.nFP), kd(f.nFP);
    std::vector<double> q(f.nFP);
    for (int i = 0; i < f.nFP; ++i)
    {
        s0[i] = fp[i].s0;
        n[i] = fp[i].n;
        kd[i] = (fp[i].mass ? 1 : 0) | (fp[i].outlet ? 2 : 0);
        q[i] = fp[i].flowRate;
    }
    f.fpS0 = to_dev(s0);
    f.fpN = to_dev(n);
    f.fpKind = to_dev(kd);
    f.fpQ = to_dev(q);
}

void Engine::fputFlowRate(int id, int fpIndex, double flowRate)
{
    Flow& f = p_->flow(id);
    if (fpIndex < 0 || fpIndex >= f.nFP) fatal("Engine::fputFlowRate: bad index %d", fpIndex);
    thrust::copy(&flowRate, &flowRate + 1, f.fpQ.begin() + fpIndex);
}

void Engine::fputSpecie(int id, int m, const std::vector<double>& Y, const std::vector<int>& yKind,
                        const std::vector<double>& yVal, const std::vector<double>& yGrad,
                        const std::vector<double>& yVf)
{
    Flow& f = p_->flow(id);
    if (m < 0 || m >= f.nSp) fatal("Engine::fputSpecie: bad specie index %d", m);
    if (long(Y.size()) != f.nc || long(yKind.size()) != f.nS || long(yVal.size()) != f.nS
     || long(yGrad.size()) != f.nS || long(yVf.size()) != f.nS)
        fatal("Engine::fputSpecie: size mismatch");
    thrust::copy(Y.begin(), Y.end(), f.Y.begin() + long(m)*f.nc);
    thrust::copy(Y.begin(), Y.end(), f.Y0.begin() + long(m)*f.nc);
    const long s0 = long(m)*f.nS;
    thrust::copy(yKind.begin(), yKind.end(), f.yKind.begin() + s0);
    thrust::copy(yVal.begin(), yVal.end(), f.yVal.begin() + s0);
    thrust::copy(yGrad.begin(), yGrad.end(), f.yGrad.begin() + s0);
    thrust::copy(yVf.begin(), yVf.end(), f.yVf.begin() + s0);
}

void Engine::fputSpecieVal(int id, int m, const std::vector<double>& yVal, long slotOffset)
{
    Flow& f = p_->flow(id);
    if (m < 0 || m >= f.nSp || slotOffset < 0 || slotOffset + long(yVal.size()) > f.nS)
        fatal("Engine::fputSpecieVal: range out of bounds");
    thrust::copy(yVal.begin(), yVal.end(), f.yVal.begin() + long(m)*f.nS + slotOffset);
}

void Engine::fputSpecieSource(int id, int m, const std::vector<double>& Su, const std::vector<double>& Sp)
{
    Flow& f = p_->flow(id);
    if (m < 0 || m >= f.nSp || long(Su.size()) != f.nc || long(Sp.size()) != f.nc)
        fatal("Engine::fputSpecieSource: size mismatch");
    thrust::copy(Su.begin(), Su.end(), f.SuY.begin() + long(m)*f.nc);
    thrust::copy(Sp.begin(), Sp.end(), f.SpY.begin() + long(m)*f.nc);
}

void Engine::fgetSpecie(int id, int m, std::vector<double>& Y, std::vector<double>& Yb) const
{
    const Flow& f = p_->flow(id);
    if (m < 0 || m >= f.nSp) fatal("Engine::fgetSpecie: bad specie index %d", m);
    Y.resize(f.nc);
    Yb.resize(f.nS);
    const long c0 = long(m)*f.nc, s0 = long(m)*f.nS;
    thrust::copy(f.Y.begin() + c0, f.Y.begin() + c0 + f.nc, Y.begin());
    thrust::copy(f.Yb.begin() + s0, f.Yb.begin() + s0 + f.nS, Yb.begin());
}

void Engine::fluidUpdateMixture(int id)
{
    p_->flow(id).updateMixture(true);
}

void Engine::fputTurb(int id, const std::vector<double>& k, const std::vector<double>& eps,
                      const std::vector<double>& nut, const std::vector<double>& alphat,
                      const std::vector<int>& kKind, const std::vector<double>& kVal,
                      const std::vector<int>& epsKind, const std::vector<double>& epsVal,
                      const std::vector<int>& nutKind, const std::vector<int>& atKind,
                      const std::vector<double>& eCmu, const std::vector<double>& eKappa,
                      const std::vector<double>& nCmu, const std::vector<double>& nKappa,
                      const std::vector<double>& nE, const std::vector<double>& atPrt,
                      const std::vector<double>& atKappa, const std::vector<double>& atE,
                      const std::vector<double>& sY, const std::vector<double>& alphatB0)
{
    Flow& f = p_->flow(id);
    if (long(k.size()) != f.nc || long(eps.size()) != f.nc || long(nut.size()) != f.nc || long(alphat.size()) != f.nc
     || long(kKind.size()) != f.nS || long(kVal.size()) != f.nS || long(epsKind.size()) != f.nS
     || long(epsVal.size()) != f.nS || long(nutKind.size()) != f.nS || long(atKind.size()) != f.nS
     || long(eCmu.size()) != f.nS || long(eKappa.size()) != f.nS || long(nCmu.size()) != f.nS
     || long(nKappa.size()) != f.nS || long(nE.size()) != f.nS || long(atPrt.size()) != f.nS
     || long(atKappa.size()) != f.nS || long(atE.size()) != f.nS || long(sY.size()) != f.nS
     || long(alphatB0.size()) != f.nS)
        fatal("Engine::fputTurb: size mismatch");
    f.setupTurbulence(k, eps, nut, alphat, kKind, kVal, epsKind, epsVal, nutKind, atKind, eCmu, eKappa,
                      nCmu, nKappa, nE, atPrt, atKappa, atE, sY, alphatB0);
}

void Engine::fluidSpecies(int id, double dt, const std::vector<double>& alphaY, const SolverCtl& ctl,
                          std::vector<SolveInfo>& info)
{
    p_->flow(id).species(dt, alphaY, ctl, info);
}

void Engine::fluidStart(int id, bool phiGiven)
{
    Flow& f = p_->flow(id);
    f.updateRho();
    f.updateBoundary(false);
    if (!phiGiven) f.initPhi();
    f.updateBoundary(false);
    f.updatePAbs();
    f.validateTurbulence();
}

void Engine::fluidStoreOld(int id, double dt, bool frozen, bool storePrevIter)
{
    p_->flow(id).storeOld(dt, frozen, storePrevIter);
}

void Engine::fluidMomentum(int id, double dt, double alphaU, const SolverCtl& ctl, bool predictor, SolveInfo info[3])
{
    p_->flow(id).momentum(dt, alphaU, ctl, predictor, info);
}

SolveInfo Engine::fluidPressure(int id, double dt, const SolverCtl& ctl, double pRelax, int nNonOrth, double contErr[2])
{
    SolveInfo info;
    p_->flow(id).pressure(dt, ctl, pRelax, nNonOrth, info, contErr);
    return info;
}

void Engine::fluidTurbulence(int id, double dt, double alphaK, double alphaEps, const SolverCtl& ctlK,
                             const SolverCtl& ctlEps, SolveInfo& infoEps, SolveInfo& infoK)
{
    p_->flow(id).turbulence(dt, alphaK, alphaEps, ctlK, ctlEps, infoEps, infoK);
}

void Engine::fluidUpdateRho(int id)
{
    p_->flow(id).updateRho();
}

void Engine::fluidUpdateBoundary(int id, bool evalFlowPatches)
{
    p_->flow(id).updateBoundary(evalFlowPatches);
}

void Engine::fluidToEnergy(int id, double dt, bool dpdt)
{
    p_->flow(id).toEnergy(dt, dpdt);
}

void Engine::fluidRhoToEnergy(int id)
{
    Flow& f = p_->flow(id);
    thrust::copy(f.rho.begin(), f.rho.end(), p_->E.rho.begin() + f.c0);
}

double Engine::fluidCourantNo(int id, double dt, double& meanCo) const
{
    return p_->flow(id).courantNo(dt, meanCo);
}

void Engine::buildingInit(const BuildingIn& bi, int& nGlaze, int& nGlazeMiss, double& areaExt, double& areaGlaze)
{
    p_->bld.init(p_->m, bi, nGlaze, nGlazeMiss, areaExt, areaGlaze);
}

void Engine::buildingUpdate(double t, const std::vector<double>& zoneQrad)
{
    p_->bld.updateEnergy(p_->E, t, zoneQrad);
}

WeatherState Engine::buildingWeather(double t) const
{
    return p_->bld.weather.at(t);
}

} // namespace gc
