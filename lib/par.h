// Backend layer: the same source compiles with nvcc (CUDA device) or with g++ -fopenmp (Thrust OMP "device").
// Kernels are lambdas over an index range; they capture only local copies (raw device pointers, scalars),
// never `this`, so the same code runs on both backends.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/for_each.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

#if defined(__CUDACC__)
#define HD __host__ __device__
#define GC_CUDA 1
#else
#define HD
#endif

namespace gc
{

using label = int;                         // cell / face index (OpenFOAM Int32 build, meshes < 2^31)
template<class T> using dvec = thrust::device_vector<T>;
template<class T> using hvec = thrust::host_vector<T>;

template<class T> inline T* ptr(dvec<T>& v) { return thrust::raw_pointer_cast(v.data()); }
template<class T> inline const T* ptr(const dvec<T>& v) { return thrust::raw_pointer_cast(v.data()); }

// for i in [0, n): f(i)
template<class F> inline void for_n(long n, F f)
{
    if (n > 0)
        thrust::for_each_n(thrust::device, thrust::counting_iterator<long>(0), n, f);
}

// sum / max / min over i in [0, n) of f(i)
template<class F> inline double sum_n(long n, F f)
{
    if (n <= 0) return 0.0;
    return thrust::transform_reduce(thrust::device, thrust::counting_iterator<long>(0),
                                    thrust::counting_iterator<long>(n), f, 0.0, thrust::plus<double>());
}

template<class F> inline double max_n(long n, F f, double init = -1e300)
{
    if (n <= 0) return init;
    return thrust::transform_reduce(thrust::device, thrust::counting_iterator<long>(0),
                                    thrust::counting_iterator<long>(n), f, init, thrust::maximum<double>());
}

template<class F> inline double min_n(long n, F f, double init = 1e300)
{
    if (n <= 0) return init;
    return thrust::transform_reduce(thrust::device, thrust::counting_iterator<long>(0),
                                    thrust::counting_iterator<long>(n), f, init, thrust::minimum<double>());
}

template<class T> inline void fill(dvec<T>& v, T x) { thrust::fill(thrust::device, v.begin(), v.end(), x); }

template<class T> inline void copy(const dvec<T>& a, dvec<T>& b)
{
    b.resize(a.size());
    thrust::copy(thrust::device, a.begin(), a.end(), b.begin());
}

template<class T> inline dvec<T> to_dev(const std::vector<T>& h) { return dvec<T>(h.begin(), h.end()); }

template<class T> inline std::vector<T> to_host(const dvec<T>& d)
{
    std::vector<T> h(d.size());
    thrust::copy(d.begin(), d.end(), h.begin());
    return h;
}

void sync();                               // device synchronise (no-op on the OMP backend)
double mem_gb();                           // device memory in use [GB] (CUDA) / process RSS (OMP)

[[noreturn]] void fatal(const char* fmt, ...);

} // namespace gc
