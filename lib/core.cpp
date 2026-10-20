#include "par.h"

#include <cstdarg>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#if defined(GC_CUDA)
#include <cuda_runtime.h>
#endif

namespace gc
{

void sync()
{
#if defined(GC_CUDA)
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) fatal("CUDA error: %s", cudaGetErrorString(e));
#endif
}

double mem_gb()
{
#if defined(GC_CUDA)
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    return double(tot - fr)/(1024.0*1024.0*1024.0);
#else
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.compare(0, 6, "VmRSS:") == 0) return std::stod(line.substr(6))/(1024.0*1024.0);
    return -1.0;
#endif
}

// ---------------------------------------------------------------------------- phase profiler
namespace
{
struct ProfData
{
    bool on = false;
    std::vector<std::string> stack;
    std::map<std::string, std::pair<double, long>> acc;   // path -> (seconds, calls)
    ProfData()
    {
        const char* e = std::getenv("GPUCHT_PROF");
        on = e && *e && std::string(e) != "0";
    }
};
ProfData& prof()
{
    static ProfData d;
    return d;
}
} // namespace

bool prof_on()
{
    return prof().on;
}

ProfScope::ProfScope(const char* name) : on_(prof().on)
{
    if (!on_) return;
    sync();
    ProfData& d = prof();
    d.stack.push_back(d.stack.empty() ? std::string(name) : d.stack.back() + "/" + name);
    t0_ = std::chrono::steady_clock::now();
}

ProfScope::~ProfScope()
{
    if (!on_) return;
    sync();
    ProfData& d = prof();
    auto& a = d.acc[d.stack.back()];
    a.first += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
    a.second += 1;
    d.stack.pop_back();
}

std::string prof_report()
{
    std::string s;
    char buf[256];
    for (const auto& kv : prof().acc)
    {
        std::snprintf(buf, sizeof(buf), "    %-44s %10.4f s %8ld calls %10.3f ms/call\n", kv.first.c_str(),
                      kv.second.first, kv.second.second, 1e3*kv.second.first/double(kv.second.second));
        s += buf;
    }
    return s;
}

void fatal(const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "\n--> gpucht FATAL ERROR: ");
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
    std::fflush(stderr);
    std::abort();
}

} // namespace gc
