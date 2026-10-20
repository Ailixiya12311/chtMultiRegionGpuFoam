// Building boundary physics (cht/building/ port): design-day weather, depth-map solar shading with glazing
// transmission, interior longwave star-node radiation, and exterior Robin (gpuCht::exteriorSurface) updates.
#pragma once

#include "energy.h"

namespace gc
{

constexpr double SIGMA_SB = 5.670374419e-8;

struct Weather
{
    std::vector<double> t, Ta, Tsky, p, w, wind, DNI, DHI, GHI;
    std::vector<v3> sun;
    double rhoGround = 0.2;

    void load(const std::string& path, double rhoG);
    WeatherState at(double time) const;
};

struct Samples
{
    int n = 0;                                // number of faces
    std::vector<v3> P;                        // sample points in 3-D
    std::vector<int> face;                    // parent face index [0, n) of each sample
    std::vector<double> count;                // samples per face

    void build(const std::vector<v3>& Cf, const std::vector<v3>& nrm, const std::vector<double>& A,
               double pixel, int maxM = 4);
};

struct ShadowGroup
{
    const Samples* S = nullptr;
    const std::vector<v3>* nrm = nullptr;
    bool glass = false;                       // false: opaque, true: glass
    std::vector<double> vis, glf;             // output per face: visible fraction, fraction with glass in front
};

struct ShadowMap
{
    double pixel = 0.05;
    void compute(v3 sunDir, std::vector<ShadowGroup>& groups) const;
};

struct Building
{
    const DMesh* m = nullptr;
    BuildingIn cfg;
    Weather weather;

    int nE = 0, nI_ = 0, nz = 0;
    // host geometry for solar / shadow-map updates
    std::vector<v3> eC, eN, eN_opq, eN_gl;
    std::vector<double> eA, eA_opq, eA_gl;
    std::vector<int> eZone;
    std::vector<v3> iC, iN, iN_opq;
    std::vector<double> iA, iA_opq;
    std::vector<double> zoneA, zoneAOpq;
    Samples sampExt, sampGl, sampInt;
    ShadowMap smap;
    bool hasSolar = false;
    double solarT = -1e300;
    WeatherState curW;

    // device arrays for per-outer-corrector updates
    dvec<int> eb;
    dvec<double> eHc, eEps, eNz, absorbedE;
    dvec<int> iPos, iZone;
    dvec<char> iAirOwner, iGlass;
    dvec<double> iA_d, iEps, absorbedI, Ts;
    dvec<int> zPtr, zFace;
    dvec<double> zoneA_d, zoneQrad_d, Tz3_d, Tmrt_d;

    void init(const DMesh& mesh, const BuildingIn& bi, int& nGlaze, int& nGlazeMiss,
              double& areaExt, double& areaGlaze);
    void updateSolar(double t);
    void updateEnergy(Energy& E, double t, const std::vector<double>& zoneQrad);
};

} // namespace gc
