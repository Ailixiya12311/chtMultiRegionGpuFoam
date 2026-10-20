/*---------------------------------------------------------------------------*\
Application
    chtMultiRegionGpuFoam

Description
    chtMultiRegionFoam (OpenFOAM v2606) with the equations assembled and solved by the gpucht library on a GPU
    (CUDA) or on CPU threads (OpenMP build). OpenFOAM reads the case (meshes, fields, boundary conditions,
    dictionaries) and writes the results; the library holds the fields during the run.

    Energy: one monolithic system over all regions (the coupled interfaces are part of the matrix), equal at
    convergence to OpenFOAM's region-by-region coupling.
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "regionProperties.H"
#include "fvSolution.H"

#include "regions.H"
#include "energyCoupling.H"
#include "fluidCoupling.H"
#include "api.h"

#include <chrono>

using namespace Foam;

// wall time per phase of the time loop (device synchronised at each mark), summarised at the end
class PhaseTimer
{
    const gc::Engine& e_;
    std::chrono::steady_clock::time_point t0_ = std::chrono::steady_clock::now();
    DynamicList<word> names_;
    DynamicList<scalar> secs_;

public:
    explicit PhaseTimer(const gc::Engine& e) : e_(e) {}

    void reset()
    {
        e_.sync();
        t0_ = std::chrono::steady_clock::now();
    }

    // time since the previous mark goes to phase `name`
    void mark(const word& name)
    {
        e_.sync();
        const auto t = std::chrono::steady_clock::now();
        const scalar s = std::chrono::duration<double>(t - t0_).count();
        t0_ = t;
        label i = names_.find(name);
        if (i < 0)
        {
            names_.append(name);
            secs_.append(0);
            i = names_.size() - 1;
        }
        secs_[i] += s;
    }

    void report(label nSteps) const
    {
        scalar total = 0;
        forAll(secs_, i) total += secs_[i];
        Info<< "Time loop wall time by phase (" << nSteps << " steps, " << total << " s):" << nl;
        forAll(names_, i)
        {
            Info<< "    " << names_[i].c_str() << ": " << secs_[i] << " s, "
                << secs_[i]/max(nSteps, label(1)) << " s/step, " << 100*secs_[i]/max(total, VSMALL) << " %" << nl;
        }
        Info<< endl;
    }
};

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "chtMultiRegionFoam with the equations solved by the gpucht library (CUDA or OpenMP)"
    );
    argList::noParallel();

    #include "setRootCase.H"
    #include "createTime.H"

    regionProperties rp(runTime);
    const wordList fluidNames(rp.found("fluid") ? rp["fluid"] : wordList());
    const wordList solidNames(rp.found("solid") ? rp["solid"] : wordList());

    PtrList<fvMesh> meshes(fluidNames.size() + solidNames.size());
    List<gpuCht::Region> regions(meshes.size());
    {
        label k = 0;
        for (const bool fluid : {true, false})
        {
            for (const word& name : (fluid ? fluidNames : solidNames))
            {
                Info<< "Create " << (fluid ? "fluid" : "solid") << " mesh for region " << name
                    << " for time = " << runTime.timeName() << nl << endl;
                meshes.set(k, new fvMesh(IOobject(name, runTime.timeName(), runTime, IOobject::MUST_READ)));
                regions[k].name = name;
                regions[k].fluid = fluid;
                regions[k].mesh = &meshes[k];
                ++k;
            }
        }
    }

    const bool adjustTimeStep = runTime.controlDict().getOrDefault<bool>("adjustTimeStep", false);
    const scalar maxCo = runTime.controlDict().getOrDefault<scalar>("maxCo", 1.0);
    const scalar maxDi = runTime.controlDict().getOrDefault<scalar>("maxDi", 10.0);
    const scalar maxDeltaT = runTime.controlDict().getOrDefault<scalar>("maxDeltaT", GREAT);

    gc::MeshIn mi;
    gpuCht::buildGlobalMesh(regions, mi);
    gc::Engine engine(mi);
    {
        auto drop = [](std::vector<double>& v) { std::vector<double>().swap(v); };
        drop(mi.V); drop(mi.C); drop(mi.Sf); drop(mi.Cf); drop(mi.w);
        drop(mi.nonOrthDC); drop(mi.corrVec); drop(mi.lsP); drop(mi.lsN);
        drop(mi.dcI_A); drop(mi.dcI_B); drop(mi.dcB); drop(mi.lsB);
    }
    Info<< "device memory after the mesh: " << engine.memGB() << " GB" << nl << endl;

    const fvSolution solutionDict(runTime);
    const dictionary& pimple = solutionDict.subDict("PIMPLE");
    const label nOuterCorr = pimple.getOrDefault<label>("nOuterCorrectors", 1);
    const dictionary* ocrc = pimple.findDict("outerCorrectorResidualControl");

    gpuCht::EnergyCoupling energy(runTime, regions, mi, engine);
    gpuCht::FluidCoupling fluids(runTime, regions, mi, engine, energy, nOuterCorr);
    Info<< "device memory after the fields: " << engine.memGB() << " GB" << nl << endl;

    // compressibleMultiRegionCourantNo.H and solidRegionDiffusionNo.H run (and print) whether or not the time step
    // is adjusted; then setInitialMultiRegionDeltaT.H
    {
        const scalar dt0 = runTime.deltaTValue();
        scalar meanCo = 0, meanDi = 0;
        scalar CoNum = fluids.courantNo(dt0, meanCo);
        scalar DiNum = energy.solidDiffNo(dt0, meanDi);
        if (adjustTimeStep && (runTime.timeIndex() == 0) && ((CoNum > SMALL) || (DiNum > SMALL)))
        {
            if (CoNum < SMALL) CoNum = SMALL;
            if (DiNum < SMALL) DiNum = SMALL;
            runTime.setDeltaT
            (
                min
                (
                    min(maxCo/CoNum, maxDi/DiNum)*dt0,
                    min(dt0, maxDeltaT)
                )
            );
            Info<< "deltaT = " << runTime.deltaTValue() << endl;
        }
    }

    Info<< "\nStarting time loop\n" << endl;

    PhaseTimer timer(engine);
    label nSteps = 0;
    timer.reset();

    while (runTime.run())
    {
        timer.mark("functionObjects");

        // compressibleMultiRegionCourantNo.H, solidRegionDiffusionNo.H (always), setMultiRegionDeltaT.H
        const scalar dt0 = runTime.deltaTValue();
        scalar meanCo = 0, meanDi = 0;
        scalar CoNum = fluids.courantNo(dt0, meanCo);
        scalar DiNum = energy.solidDiffNo(dt0, meanDi);
        if (adjustTimeStep)
        {
            if (CoNum == -GREAT) CoNum = SMALL;
            if (DiNum == -GREAT) DiNum = SMALL;
            const scalar maxDeltaTFluid = maxCo/(CoNum + SMALL);
            const scalar maxDeltaTSolid = maxDi/(DiNum + SMALL);
            const scalar deltaTFluid =
                min(min(maxDeltaTFluid, 1.0 + 0.1*maxDeltaTFluid), 1.2);
            runTime.setDeltaT
            (
                min
                (
                    min(deltaTFluid, maxDeltaTSolid)*dt0,
                    maxDeltaT
                )
            );
            Info<< "deltaT = " << runTime.deltaTValue() << endl;
        }

        ++runTime;
        const scalar dt = runTime.deltaTValue();

        Info<< "Time = " << runTime.timeName() << nl << endl;

        // old time levels: rho.oldTime() of the fluid EEqn is the density before rhoEqn
        fluids.rhoToEnergy();
        engine.energyStoreOld();
        fluids.storeOld(dt);
        fluids.hvacControl(runTime.value() - dt, dt);
        timer.mark("old time level, HVAC");

        for (label oCorr = 0; oCorr < nOuterCorr; ++oCorr)
        {
            const bool finalIter = (oCorr == nOuterCorr - 1);

            if (nOuterCorr != 1)
            {
                Info<< "PIMPLE: iteration " << oCorr + 1 << endl;
            }

            fluids.momentum(dt, finalIter);
            timer.mark("momentum");
            fluids.species(dt, finalIter);
            timer.mark("species");

            fluids.toEnergy(dt);
            energy.updateBCs();
            energy.setRelax(finalIter);
            const gc::SolverCtl ctl = energy.solverCtl(finalIter);
            const gc::SolveInfo si = engine.energySolve(dt, ctl, true);
            Info<< "\nSolving energy (all regions)" << nl
                << ctl.name.c_str() << ":  Solving for h, Initial residual = " << si.res0
                << ", Final residual = " << si.res << ", No Iterations " << si.iters << endl;
            // one system for all regions: every region reports it as its h solve
            const SolverPerformance<scalar> hPerf(ctl.name, "h", si.res0, si.res, si.iters, gpuCht::converged(ctl, si));
            forAll(meshes, i) meshes[i].data().setSolverPerformance("h", hPerf);
            energy.logMinMax();
            timer.mark("energy");

            fluids.pressure(dt, finalIter);
            timer.mark("pressure");
            fluids.turbulence(dt, finalIter);
            timer.mark("turbulence");

            if (ocrc && !finalIter && fluids.outerConverged(*ocrc, si.res0))
            {
                Info<< "PIMPLE: converged in " << oCorr + 1 << " iterations" << endl;
                break;
            }
        }

        // the device holds the fields; OpenFOAM's copies are refreshed for writing and, while there are
        // functionObjects, every time step (Time::run executes them at the start of the next step)
        const functionObjectList& fos = runTime.functionObjects();
        if (runTime.writeTime() || (fos.status() && !fos.empty()))
        {
            energy.writeBack();
            fluids.writeBack();
        }
        timer.mark("device -> OpenFOAM fields");
        runTime.write();
        timer.mark("write");
        ++nSteps;
        if (nSteps == 1)
        {
            // the linear-solver structures (AMG hierarchies, colourings, work vectors) exist from the first step on
            Info<< "device memory after the first time step: " << engine.memGB() << " GB" << nl << endl;
        }

        runTime.printExecutionTime(Info);
    }
    timer.mark("functionObjects");
    timer.report(nSteps);
    const std::string prof = engine.profReport();
    if (!prof.empty())
    {
        Info<< "Library scopes (GPUCHT_PROF, inclusive):" << nl << prof.c_str() << endl;
    }

    Info<< "End\n" << endl;

    return 0;
}
