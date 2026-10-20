#include "building.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace gc
{

namespace
{

constexpr double DAY = 86400.0;
constexpr double EPS_W = 0.621945;
constexpr double PI = 3.14159265358979323846;

double saturationPressurePa(double T_C)
{
    const double K = T_C + 273.15;
    double logp;
    if (T_C >= 0.0)
    {
        logp = -5800.2206/K + 1.3914993 - 0.048640239*K + 4.1764768e-5*K*K
             - 1.4452093e-8*K*K*K + 6.5459673*std::log(K);
    }
    else
    {
        logp = -5674.5359/K + 6.3925247 - 0.009677843*K + 6.2215701e-7*K*K
             + 2.0747825e-9*K*K*K - 9.484024e-13*K*K*K*K + 4.1635019*std::log(K);
    }
    return std::exp(logp);
}

double dewPointC(double w, double p)
{
    const double pv = w*p/(EPS_W + w);
    double lo = -60.0, hi = 80.0;
    for (int i = 0; i < 80; ++i)
    {
        const double mid = 0.5*(lo + hi);
        if (saturationPressurePa(mid) < pv) lo = mid;
        else hi = mid;
    }
    return 0.5*(lo + hi);
}

std::vector<std::string> splitCsv(const std::string& line)
{
    std::vector<std::string> out;
    std::string cur;
    for (char c : line)
    {
        if (c == '\r' || c == '\n') continue;
        if (c == ',') { out.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}

double interpPeriodic(double t, const std::vector<double>& tt, const std::vector<double>& y)
{
    const size_t n = tt.size();
    if (n == 0) return 0.0;
    t = std::fmod(std::fmod(t, DAY) + DAY, DAY);
    if (t <= tt.front())
    {
        const double t0 = tt.back() - DAY, t1 = tt.front();
        const double a = (t - t0)/std::max(t1 - t0, 1e-30);
        return y.back() + a*(y.front() - y.back());
    }
    if (t >= tt.back())
    {
        const double t0 = tt.back(), t1 = tt.front() + DAY;
        const double a = (t - t0)/std::max(t1 - t0, 1e-30);
        return y.back() + a*(y.front() - y.back());
    }
    const auto it = std::upper_bound(tt.begin(), tt.end(), t);
    const size_t i1 = size_t(it - tt.begin());
    const size_t i0 = i1 - 1;
    const double a = (t - tt[i0])/std::max(tt[i1] - tt[i0], 1e-30);
    return y[i0] + a*(y[i1] - y[i0]);
}

void tangents(v3 n, v3& t1, v3& t2)
{
    const v3 ref = std::fabs(n.z) < 0.9 ? V3(0, 0, 1) : V3(1, 0, 0);
    t1 = V3(ref.y*n.z - ref.z*n.y, ref.z*n.x - ref.x*n.z, ref.x*n.y - ref.y*n.x);
    t1 = t1*(1.0/mag(t1));
    t2 = V3(n.y*t1.z - n.z*t1.y, n.z*t1.x - n.x*t1.z, n.x*t1.y - n.y*t1.x);
}

v3 cross3(v3 a, v3 b)
{
    return V3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

int64_t gridKey(v3 p)
{
    const int64_t q0 = static_cast<int64_t>(std::rint(p.x*1000.0)) + (1LL << 20);
    const int64_t q1 = static_cast<int64_t>(std::rint(p.y*1000.0)) + (1LL << 20);
    const int64_t q2 = static_cast<int64_t>(std::rint(p.z*1000.0)) + (1LL << 20);
    return (q0 << 42) + (q1 << 21) + q2;
}

} // namespace

// ---------------------------------------------------------------------------- Weather
void Weather::load(const std::string& path, double rhoG)
{
    rhoGround = rhoG;
    std::ifstream in(path);
    if (!in) fatal("Weather: cannot open %s", path.c_str());
    std::string hdr;
    if (!std::getline(in, hdr)) fatal("Weather: empty file %s", path.c_str());
    if (hdr.size() >= 3 && static_cast<unsigned char>(hdr[0]) == 0xEF
        && static_cast<unsigned char>(hdr[1]) == 0xBB && static_cast<unsigned char>(hdr[2]) == 0xBF)
    {
        hdr.erase(0, 3);
    }
    const auto cols = splitCsv(hdr);
    std::unordered_map<std::string, size_t> colIdx;
    for (size_t i = 0; i < cols.size(); ++i) colIdx[cols[i]] = i;
    auto reqCol = [&](const char* name) -> size_t {
        auto it = colIdx.find(name);
        if (it == colIdx.end()) fatal("Weather: column %s missing in %s", name, path.c_str());
        return it->second;
    };
    const size_t cT = reqCol("sample_midpoint_s");
    const size_t cTa = reqCol("dry_bulb_C");
    const size_t cP = reqCol("pressure_Pa");
    const size_t cW = reqCol("humidity_ratio_kg_kg_dry");
    const size_t cWind = reqCol("wind_speed_design_m_s");
    const size_t cDNI = reqCol("DNI_W_m2");
    const size_t cDHI = reqCol("DHI_W_m2");
    const size_t cGHI = reqCol("GHI_W_m2");
    const size_t cSx = reqCol("towards_sun_x_east");
    const size_t cSy = reqCol("towards_sun_y_plan_north");
    const size_t cSz = reqCol("towards_sun_z_up");

    t.clear(); Ta.clear(); Tsky.clear(); p.clear(); w.clear(); wind.clear();
    DNI.clear(); DHI.clear(); GHI.clear(); sun.clear();
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() || (line.size() == 1 && line[0] == '\r')) continue;
        const auto f = splitCsv(line);
        auto val = [&](size_t idx) -> double {
            return (idx < f.size() && !f[idx].empty()) ? std::stod(f[idx]) : 0.0;
        };
        const double ti = val(cT);
        const double tai = val(cTa) + 273.15;
        const double pi = val(cP);
        const double wi = val(cW);
        const double tdp = dewPointC(wi, pi);
        const double h = ti/3600.0;
        const double eps = 0.711 + 0.56*(tdp/100.0) + 0.73*(tdp/100.0)*(tdp/100.0) + 0.013*std::cos(2.0*PI*h/24.0);
        t.push_back(ti);
        Ta.push_back(tai);
        Tsky.push_back(std::pow(eps, 0.25)*tai);
        p.push_back(pi);
        w.push_back(wi);
        wind.push_back(val(cWind));
        DNI.push_back(val(cDNI));
        DHI.push_back(val(cDHI));
        GHI.push_back(val(cGHI));
        sun.push_back(V3(val(cSx), val(cSy), val(cSz)));
    }
}

WeatherState Weather::at(double time) const
{
    WeatherState ws;
    ws.t = time;
    ws.Ta = interpPeriodic(time, t, Ta);
    ws.Tsky = interpPeriodic(time, t, Tsky);
    ws.p = interpPeriodic(time, t, p);
    ws.w = interpPeriodic(time, t, w);
    ws.wind = interpPeriodic(time, t, wind);
    ws.DNI = std::max(0.0, interpPeriodic(time, t, DNI));
    ws.DHI = std::max(0.0, interpPeriodic(time, t, DHI));
    ws.GHI = std::max(0.0, interpPeriodic(time, t, GHI));
    ws.rhoGround = rhoGround;
    std::vector<double> sx(sun.size()), sy(sun.size()), sz(sun.size());
    for (size_t i = 0; i < sun.size(); ++i) { sx[i] = sun[i].x; sy[i] = sun[i].y; sz[i] = sun[i].z; }
    v3 s = V3(interpPeriodic(time, t, sx), interpPeriodic(time, t, sy), interpPeriodic(time, t, sz));
    const double n = mag(s);
    if (n < 1e-9 || s.z <= 0.0)
    {
        ws.sun[0] = ws.sun[1] = ws.sun[2] = 0.0;
        ws.DNI = 0.0;
    }
    else
    {
        ws.sun[0] = s.x/n;
        ws.sun[1] = s.y/n;
        ws.sun[2] = s.z/n;
    }
    return ws;
}

// ---------------------------------------------------------------------------- Samples & ShadowMap
void Samples::build(const std::vector<v3>& Cf, const std::vector<v3>& nrm, const std::vector<double>& A,
                    double pixel, int maxM)
{
    n = int(A.size());
    P.clear();
    face.clear();
    count.assign(static_cast<size_t>(n), 0.0);
    if (n == 0) return;

    std::vector<int> m(static_cast<size_t>(n));
    std::vector<double> side(static_cast<size_t>(n));
    std::vector<v3> t1(static_cast<size_t>(n)), t2(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        side[i] = std::sqrt(A[i]);
        const int mm = static_cast<int>(std::ceil(side[i]/pixel));
        m[i] = std::max(1, std::min(maxM, mm));
        tangents(nrm[i], t1[i], t2[i]);
    }
    for (int mm = 1; mm <= maxM; ++mm)
    {
        for (int i = 0; i < n; ++i)
        {
            if (m[i] != mm) continue;
            const double s = side[i];
            for (int ia = 0; ia < mm; ++ia)
            {
                const double a = (ia + 0.5)/mm - 0.5;
                for (int ib = 0; ib < mm; ++ib)
                {
                    const double b = (ib + 0.5)/mm - 0.5;
                    P.push_back(Cf[i] + (t1[i]*a + t2[i]*b)*s);
                    face.push_back(i);
                }
            }
            count[i] = double(mm*mm);
        }
    }
}

void ShadowMap::compute(v3 s, std::vector<ShadowGroup>& groups) const
{
    const double ps = pixel;
    const v3 ref = std::fabs(s.z) < 0.9 ? V3(0, 0, 1) : V3(1, 0, 0);
    v3 e1 = cross3(s, ref);
    e1 = e1*(1.0/mag(e1));
    const v3 e2 = cross3(s, e1);

    struct Prep
    {
        std::vector<int> fk;
        std::vector<double> u, v, d, cs;
        std::vector<int> pix;
    };
    std::vector<Prep> prep(groups.size());
    double umin = 1e300, vmin = 1e300, umax = -1e300, vmax = -1e300;
    for (size_t g = 0; g < groups.size(); ++g)
    {
        const Samples& S = *groups[g].S;
        const std::vector<v3>& nrm = *groups[g].nrm;
        groups[g].vis.assign(static_cast<size_t>(S.n), 0.0);
        groups[g].glf.assign(static_cast<size_t>(S.n), 0.0);
        if (S.P.empty()) continue;
        std::vector<double> faceCs(static_cast<size_t>(S.n));
        for (int f = 0; f < S.n; ++f) faceCs[f] = dot(nrm[f], s);
        Prep& pr = prep[g];
        for (size_t k = 0; k < S.P.size(); ++k)
        {
            const int f = S.face[k];
            const double cs = faceCs[f];
            if (!(cs > 1e-6)) continue;
            const v3 p = S.P[k];
            const double u = dot(p, e1), v = dot(p, e2), d = -dot(p, s);
            pr.fk.push_back(f);
            pr.u.push_back(u);
            pr.v.push_back(v);
            pr.d.push_back(d);
            pr.cs.push_back(cs);
            umin = std::min(umin, u); umax = std::max(umax, u);
            vmin = std::min(vmin, v); vmax = std::max(vmax, v);
        }
    }
    if (umin > umax) return;
    const int nx = static_cast<int>((umax - umin)/ps) + 2;
    const int ny = static_cast<int>((vmax - vmin)/ps) + 2;
    const size_t npix = static_cast<size_t>(nx)*static_cast<size_t>(ny);
    std::vector<double> zop(npix, 1e300), zgl(npix, 1e300);
    for (size_t g = 0; g < groups.size(); ++g)
    {
        Prep& pr = prep[g];
        pr.pix.resize(pr.u.size());
        std::vector<double>& zbuf = groups[g].glass ? zgl : zop;
        for (size_t k = 0; k < pr.u.size(); ++k)
        {
            const int ix = static_cast<int>(std::floor((pr.u[k] - umin)/ps));
            const int iy = static_cast<int>(std::floor((pr.v[k] - vmin)/ps));
            const int px = ix*ny + iy;
            pr.pix[k] = px;
            if (pr.d[k] < zbuf[px]) zbuf[px] = pr.d[k];
        }
    }
    for (size_t g = 0; g < groups.size(); ++g)
    {
        const Samples& S = *groups[g].S;
        const Prep& pr = prep[g];
        if (pr.fk.empty()) continue;
        std::vector<double>& visF = groups[g].vis;
        std::vector<double>& glF = groups[g].glf;
        for (size_t k = 0; k < pr.fk.size(); ++k)
        {
            const double cs = pr.cs[k];
            const double tan = std::sqrt(std::max(1.0 - cs*cs, 0.0))/cs;
            const double bias = ps*(1.0 + std::min(tan, 10.0)) + 0.01;
            const int px = pr.pix[k];
            const bool vis = pr.d[k] <= zop[px] + bias;
            const bool gl = zgl[px] < pr.d[k] - bias;
            const int f = pr.fk[k];
            if (vis) visF[f] += 1.0;
            if (vis && gl) glF[f] += 1.0;
        }
        for (int f = 0; f < S.n; ++f)
        {
            const double c = std::max(S.count[f], 1.0);
            visF[f] /= c;
            glF[f] /= c;
        }
    }
}

// ---------------------------------------------------------------------------- Building
void Building::init(const DMesh& mesh, const BuildingIn& bi, int& nGlaze, int& nGlazeMiss,
                    double& areaExt, double& areaGlaze)
{
    m = &mesh;
    cfg = bi;
    weather.load(bi.weatherFile, bi.groundReflectance);
    nE = int(bi.eb.size());
    nI_ = int(bi.iPos.size());
    nz = bi.nZones;
    const int nI = mesh.nI;
    const int nIr = mesh.nIr;
    const std::vector<double> Sf_h = to_host(mesh.Sf);
    const std::vector<double> Cf_h = to_host(mesh.Cf);

    eC.resize(static_cast<size_t>(nE)); eN.resize(static_cast<size_t>(nE)); eA.resize(static_cast<size_t>(nE));
    std::vector<double> eNz_h(static_cast<size_t>(nE));
    std::vector<v3> eC_opq, eC_gl;
    eN_opq.clear(); eN_gl.clear(); eA_opq.clear(); eA_gl.clear();
    areaExt = 0.0; areaGlaze = 0.0;
    for (int k = 0; k < nE; ++k)
    {
        const long f = long(nI) + bi.eb[k];
        const v3 sf = V3(Sf_h[3*f], Sf_h[3*f + 1], Sf_h[3*f + 2]);
        const double a = mag(sf);
        const v3 n = sf*(1.0/a);
        const v3 c = V3(Cf_h[3*f], Cf_h[3*f + 1], Cf_h[3*f + 2]);
        eA[k] = a; eN[k] = n; eC[k] = c; eNz_h[k] = n.z;
        areaExt += a;
        if (bi.eGlass[k])
        {
            areaGlaze += a;
            eC_gl.push_back(c); eN_gl.push_back(n); eA_gl.push_back(a);
        }
        else
        {
            eC_opq.push_back(c); eN_opq.push_back(n); eA_opq.push_back(a);
        }
    }

    iC.resize(static_cast<size_t>(nI_)); iN.resize(static_cast<size_t>(nI_)); iA.resize(static_cast<size_t>(nI_));
    std::vector<double> iEps_h(static_cast<size_t>(nI_));
    std::vector<v3> iC_opq;
    iN_opq.clear(); iA_opq.clear();
    zoneA.assign(static_cast<size_t>(nz), 0.0);
    zoneAOpq.assign(static_cast<size_t>(nz), 0.0);
    std::vector<int> isel;
    for (int k = 0; k < nI_; ++k)
    {
        const long f = long(nIr) + bi.iPos[k];
        const v3 sf = V3(Sf_h[3*f], Sf_h[3*f + 1], Sf_h[3*f + 2])*bi.iSign[k];
        const double a = mag(sf);
        const v3 n = sf*(1.0/a);
        const v3 c = V3(Cf_h[3*f], Cf_h[3*f + 1], Cf_h[3*f + 2]);
        iA[k] = a; iN[k] = n; iC[k] = c;
        iEps_h[k] = bi.iGlass[k] ? bi.glassEmissivity : bi.emissivity;
        const int z = bi.iZone[k];
        zoneA[z] += a;
        if (bi.iGlass[k])
        {
            isel.push_back(k);
        }
        else
        {
            zoneAOpq[z] += a;
            iC_opq.push_back(c); iN_opq.push_back(n); iA_opq.push_back(a);
        }
    }

    // glazing outer face -> zone of the glass-air face behind it (5 mm along -n, 1 mm grid lookup)
    eZone.assign(static_cast<size_t>(nE), -1);
    std::vector<int> gsel;
    for (int k = 0; k < nE; ++k) if (bi.eGlass[k]) gsel.push_back(k);
    nGlaze = int(gsel.size());
    nGlazeMiss = 0;
    if (!gsel.empty() && !isel.empty())
    {
        std::vector<std::pair<int64_t, int>> ki(isel.size());
        for (size_t i = 0; i < isel.size(); ++i) ki[i] = {gridKey(iC[isel[i]]), isel[i]};
        std::stable_sort(ki.begin(), ki.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        const double dists[] = {0.005, 0.004, 0.006, 0.003, 0.007};
        for (double dist : dists)
        {
            bool anyTodo = false;
            for (int k : gsel)
            {
                if (eZone[k] >= 0) continue;
                anyTodo = true;
                const int64_t q = gridKey(eC[k] - eN[k]*dist);
                auto it = std::lower_bound(ki.begin(), ki.end(), q,
                                           [](const auto& a, int64_t val) { return a.first < val; });
                if (it != ki.end() && it->first == q)
                    eZone[k] = bi.iZone[it->second];
            }
            if (!anyTodo) break;
        }
        for (int k : gsel) if (eZone[k] < 0) ++nGlazeMiss;
    }

    if (bi.shading)
    {
        sampExt.build(eC_opq, eN_opq, eA_opq, bi.pixel);
        sampGl.build(eC_gl, eN_gl, eA_gl, bi.pixel);
        sampInt.build(iC_opq, iN_opq, iA_opq, bi.pixel);
        smap.pixel = bi.pixel;
    }

    // CSR of interior faces per zone (ascending face index order -> matches np.bincount addition order)
    std::vector<int> zCnt(static_cast<size_t>(nz) + 1, 0);
    for (int k = 0; k < nI_; ++k) zCnt[bi.iZone[k] + 1]++;
    for (int z = 0; z < nz; ++z) zCnt[z + 1] += zCnt[z];
    std::vector<int> zPos(zCnt.begin(), zCnt.end() - 1);
    std::vector<int> zF(static_cast<size_t>(nI_));
    for (int k = 0; k < nI_; ++k) zF[zPos[bi.iZone[k]]++] = k;

    eb = to_dev(bi.eb);
    eHc = to_dev(bi.eHc);
    eEps = to_dev(bi.eEps);
    eNz = to_dev(eNz_h);
    absorbedE.resize(nE); fill(absorbedE, 0.0);

    iPos = to_dev(bi.iPos);
    iZone = to_dev(bi.iZone);
    iAirOwner = to_dev(bi.iAirOwner);
    iGlass = to_dev(bi.iGlass);
    iA_d = to_dev(iA);
    iEps = to_dev(iEps_h);
    absorbedI.resize(nI_); fill(absorbedI, 0.0);
    Ts.resize(nI_); fill(Ts, 0.0);

    zPtr = to_dev(zCnt);
    zFace = to_dev(zF);
    zoneA_d = to_dev(zoneA);
    zoneQrad_d.resize(nz); fill(zoneQrad_d, 0.0);
    Tz3_d.resize(nz); fill(Tz3_d, 0.0);
    Tmrt_d.resize(nz); fill(Tmrt_d, 0.0);
    hasSolar = false;
}

void Building::updateSolar(double t)
{
    curW = weather.at(t);
    if (hasSolar && std::fabs(t - solarT) < cfg.solarUpdateInterval - 1e-9 && t >= solarT)
        return;

    const WeatherState& W = curW;
    const v3 s = V3(W.sun[0], W.sun[1], W.sun[2]);
    std::vector<double> diffE(static_cast<size_t>(nE)), beamE(static_cast<size_t>(nE), 0.0), beamI(static_cast<size_t>(nI_), 0.0);
    for (int k = 0; k < nE; ++k)
    {
        const double nz_e = eN[k].z;
        diffE[k] = W.DHI*0.5*(1.0 + nz_e) + W.rhoGround*W.GHI*0.5*(1.0 - nz_e);
    }
    if (W.DNI > 0.0)
    {
        std::vector<double> litE(static_cast<size_t>(nE), 1.0), litI(static_cast<size_t>(nI_), 0.0);
        if (cfg.shading)
        {
            std::vector<ShadowGroup> groups(3);
            groups[0] = {&sampExt, &eN_opq, false, {}, {}};
            groups[1] = {&sampGl, &eN_gl, true, {}, {}};
            groups[2] = {&sampInt, &iN_opq, false, {}, {}};
            smap.compute(s, groups);
            int io = 0, ig = 0;
            for (int k = 0; k < nE; ++k)
            {
                if (cfg.eGlass[k])
                {
                    litE[k] = groups[1].vis[ig++];
                }
                else
                {
                    const double vis = groups[0].vis[io], glf = groups[0].glf[io];
                    litE[k] = vis - glf + cfg.transmittance*glf;
                    ++io;
                }
            }
            int iio = 0;
            for (int k = 0; k < nI_; ++k)
            {
                if (!cfg.iGlass[k])
                {
                    const double vis = groups[2].vis[iio], glf = groups[2].glf[iio];
                    litI[k] = glf + (vis - glf)*1.0;   // _open_view = 1.0
                    ++iio;
                }
            }
        }
        for (int k = 0; k < nE; ++k)
            beamE[k] = W.DNI*std::max(dot(eN[k], s), 0.0)*litE[k];
        for (int k = 0; k < nI_; ++k)
            beamI[k] = W.DNI*std::max(dot(iN[k], s), 0.0)*litI[k]*cfg.transmittance;
    }

    std::vector<double> absE(static_cast<size_t>(nE));
    std::vector<double> trBeam(static_cast<size_t>(nz), 0.0), trDiff(static_cast<size_t>(nz), 0.0);
    for (int k = 0; k < nE; ++k)
    {
        absE[k] = cfg.eAlpha[k]*(beamE[k] + diffE[k]);
        if (cfg.eGlass[k] && eZone[k] >= 0)
        {
            const int z = eZone[k];
            trBeam[z] += cfg.transmittance*beamE[k]*eA[k];
            trDiff[z] += cfg.transmittance*diffE[k]*eA[k];
        }
    }

    std::vector<double> land(static_cast<size_t>(nI_)), landZ(static_cast<size_t>(nz), 0.0);
    for (int k = 0; k < nI_; ++k)
    {
        land[k] = beamI[k]*iA[k];
        landZ[cfg.iZone[k]] += land[k];
    }
    std::vector<double> scale(static_cast<size_t>(nz), 0.0), lost(static_cast<size_t>(nz), 0.0);
    for (int z = 0; z < nz; ++z)
    {
        if (landZ[z] > 0.0) scale[z] = trBeam[z]/std::max(landZ[z], 1e-300);
        else lost[z] = trBeam[z];
    }
    std::vector<double> absI(static_cast<size_t>(nI_)), pool(static_cast<size_t>(nz));
    for (int z = 0; z < nz; ++z) pool[z] = trDiff[z] + lost[z];
    for (int k = 0; k < nI_; ++k)
    {
        const int z = cfg.iZone[k];
        land[k] *= scale[z];
        absI[k] = cfg.interiorAbsorptance*land[k];
        pool[z] += (1.0 - cfg.interiorAbsorptance)*land[k];
    }
    std::vector<double> spread(static_cast<size_t>(nz), 0.0);
    for (int z = 0; z < nz; ++z)
        if (zoneAOpq[z] > 0.0) spread[z] = pool[z]/std::max(zoneAOpq[z], 1e-300);
    for (int k = 0; k < nI_; ++k)
        if (!cfg.iGlass[k]) absI[k] += spread[cfg.iZone[k]]*iA[k];

    absorbedE = to_dev(absE);
    absorbedI = to_dev(absI);
    solarT = t;
    hasSolar = true;
}

void Building::updateEnergy(Energy& E, double t, const std::vector<double>& zoneQrad)
{
    updateSolar(t);
    const MeshView g = view(*m);

    // 1. Interior surface heat flows (absorbed solar + longwave star-node + radiative internal gains)
    if (nI_ > 0)
    {
        const int* ip = ptr(iPos);
        const char* ao = ptr(iAirOwner);
        const double *Tc = ptr(E.T), *go = ptr(E.gA), *gn = ptr(E.gB), *Gs = ptr(E.G), *Rl = ptr(E.Rlayer);
        double* qi = ptr(E.qIface);
        double* ts = ptr(Ts);
        for_n(nI_, [=] HD (long k) {
            const int j = ip[k];
            const int f = g.nIr + j;
            const int o = g.own[f], n = g.nei[f];
            const double R = Rl[j];
            if (R == 0.0)
                ts[k] = (go[j]*Tc[o] + gn[j]*Tc[n] + qi[j])/(go[j] + gn[j]);
            else
            {
                const double Q = (Tc[o] - Tc[n])*Gs[j];
                ts[k] = ao[k] ? (Tc[o] - Q/go[j]) : (Tc[n] + Q/gn[j]);
            }
        });

        const double* absI = ptr(absorbedI);
        if (cfg.interiorRad && nz > 0)
        {
            if (long(zoneQrad.size()) == nz)
                thrust::copy(zoneQrad.begin(), zoneQrad.end(), zoneQrad_d.begin());
            const int *zp = ptr(zPtr), *zf = ptr(zFace), *iz = ptr(iZone);
            const double *ia = ptr(iA_d), *ie = ptr(iEps), *za = ptr(zoneA_d), *zq = ptr(zoneQrad_d);
            double *tz3 = ptr(Tz3_d), *tmrt = ptr(Tmrt_d);
            for_n(nz, [=] HD (long z) {
                double sAT = 0.0;
                for (int p = zp[z]; p < zp[z + 1]; ++p)
                {
                    const int k = zf[p];
                    sAT += ia[k]*ts[k];
                }
                const double Tz = sAT/dmax(za[z], 1e-300);
                const double cTz3 = Tz*Tz*Tz;
                tz3[z] = cTz3;
                double sWT = 0.0, sW = 0.0;
                for (int p = zp[z]; p < zp[z + 1]; ++p)
                {
                    const int k = zf[p];
                    const double w = 4.0*ie[k]*SIGMA_SB*cTz3*ia[k];
                    sWT += w*ts[k];
                    sW += w;
                }
                tmrt[z] = sWT/dmax(sW, 1e-300);
            });
            for_n(nI_, [=] HD (long k) {
                const int z = iz[k];
                const double w = 4.0*ie[k]*SIGMA_SB*tz3[z]*ia[k];
                const double q = absI[k] + w*(tmrt[z] - ts[k]) + (zq[z]/dmax(za[z], 1e-300))*ia[k];
                qi[ip[k]] = q;
            });
        }
        else
        {
            for_n(nI_, [=] HD (long k) { qi[ip[k]] = absI[k]; });
        }
    }

    // 2. Exterior surfaces (Robin boundary values on E.bVal / E.bVf)
    if (nE > 0)
    {
        const int* eb_p = ptr(eb);
        const double *hc_p = ptr(eHc), *eps_p = ptr(eEps), *nz_p = ptr(eNz), *absE_p = ptr(absorbedE);
        const double *TbB = ptr(E.TbB), *kB = ptr(E.kappaB);
        double *bVal = ptr(E.bVal), *bVf = ptr(E.bVf);
        const double Ta = curW.Ta, Tsky = curW.Tsky;
        const double Ta4 = Ta*Ta*Ta*Ta, Tsky4 = Tsky*Tsky*Tsky*Tsky;
        for_n(nE, [=] HD (long k) {
            const int b = eb_p[k];
            const double Tb = TbB[b];
            const double nz_k = nz_p[k];
            const double Trad = pow(0.5*(1.0 + nz_k)*Tsky4 + 0.5*(1.0 - nz_k)*Ta4, 0.25);
            const double hr = eps_p[k]*SIGMA_SB*(Trad*Trad + Tb*Tb)*(Trad + Tb);
            const double h = hc_p[k] + hr;
            const double Teq = (hc_p[k]*Ta + hr*Trad + absE_p[k])/h;
            const double kd = kB[b]*g.dcB[b];
            bVf[b] = h/(h + kd);
            bVal[b] = Teq;
        });
    }
}

} // namespace gc
