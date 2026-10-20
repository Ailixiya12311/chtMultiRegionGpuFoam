#include "par.h"

#include <cstdarg>
#include <fstream>
#include <string>

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
