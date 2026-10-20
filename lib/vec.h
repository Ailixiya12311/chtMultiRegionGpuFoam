// Small vector helpers usable in kernels. Vectors are stored as 3 consecutive doubles (a[3*i + k]).
#pragma once

#include "par.h"

namespace gc
{

struct v3
{
    double x, y, z;
};

HD inline v3 V3(double a, double b, double c) { return v3{a, b, c}; }
HD inline v3 ld3(const double* a, long i) { return v3{a[3*i], a[3*i + 1], a[3*i + 2]}; }
HD inline void st3(double* a, long i, const v3& v) { a[3*i] = v.x; a[3*i + 1] = v.y; a[3*i + 2] = v.z; }
HD inline void add3(double* a, long i, const v3& v) { a[3*i] += v.x; a[3*i + 1] += v.y; a[3*i + 2] += v.z; }
HD inline v3 operator+(const v3& a, const v3& b) { return v3{a.x + b.x, a.y + b.y, a.z + b.z}; }
HD inline v3 operator-(const v3& a, const v3& b) { return v3{a.x - b.x, a.y - b.y, a.z - b.z}; }
HD inline v3 operator-(const v3& a) { return v3{-a.x, -a.y, -a.z}; }
HD inline v3 operator*(double s, const v3& a) { return v3{s*a.x, s*a.y, s*a.z}; }
HD inline v3 operator*(const v3& a, double s) { return v3{s*a.x, s*a.y, s*a.z}; }
HD inline v3& operator+=(v3& a, const v3& b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
HD inline v3& operator-=(v3& a, const v3& b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
HD inline double dot(const v3& a, const v3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
HD inline double mag(const v3& a) { return sqrt(dot(a, a)); }
HD inline double cmp(const v3& a, int k) { return k == 0 ? a.x : (k == 1 ? a.y : a.z); }

HD inline double dmin(double a, double b) { return a < b ? a : b; }
HD inline bool nanq(double a) { return a != a; }
HD inline double dmax(double a, double b) { return a > b ? a : b; }

// symmetric tensor (xx, xy, xz, yy, yz, zz) stored as 6 doubles
HD inline v3 symmDot(const double* t, long c, const v3& v)
{
    const double* s = t + 6*c;
    return v3{s[0]*v.x + s[1]*v.y + s[2]*v.z, s[1]*v.x + s[3]*v.y + s[4]*v.z, s[2]*v.x + s[4]*v.y + s[5]*v.z};
}

// OpenFOAM SymmTensor::safeInv (inv of a symmTensorField): near-zero diagonal components (2-D cases) are padded
// with 1 before inverting and the padding removed afterwards; a singular tensor gives zero
HD inline void symmSafeInv(double* s)
{
    const double a = s[0]*s[0], b = s[3]*s[3], c = s[5]*s[5];
    const double thr = 1e-15*(a + b + c);
    const bool sx = a < thr, sy = b < thr, sz = c < thr;
    double w[6] = {s[0] + (sx ? 1 : 0), s[1], s[2], s[3] + (sy ? 1 : 0), s[4], s[5] + (sz ? 1 : 0)};
    const double det = w[0]*(w[3]*w[5] - w[4]*w[4]) - w[1]*(w[1]*w[5] - w[4]*w[2]) + w[2]*(w[1]*w[4] - w[3]*w[2]);
    if (fabs(det) < 1e-150)
    {
        for (int k = 0; k < 6; ++k) s[k] = 0.0;
        return;
    }
    const double r = 1.0/det;
    s[0] = (w[3]*w[5] - w[4]*w[4])*r - (sx ? 1 : 0);
    s[1] = (w[2]*w[4] - w[1]*w[5])*r;
    s[2] = (w[1]*w[4] - w[2]*w[3])*r;
    s[3] = (w[0]*w[5] - w[2]*w[2])*r - (sy ? 1 : 0);
    s[4] = (w[1]*w[2] - w[0]*w[4])*r;
    s[5] = (w[0]*w[3] - w[1]*w[1])*r - (sz ? 1 : 0);
}

// in-place inverse of a symmetric tensor (OpenFOAM inv(symmTensor)); returns false when singular
HD inline bool symmInv(double* s)
{
    const double xx = s[0], xy = s[1], xz = s[2], yy = s[3], yz = s[4], zz = s[5];
    const double det = xx*(yy*zz - yz*yz) - xy*(xy*zz - yz*xz) + xz*(xy*yz - yy*xz);
    if (det == 0.0) return false;
    const double r = 1.0/det;
    s[0] = (yy*zz - yz*yz)*r;
    s[1] = (xz*yz - xy*zz)*r;
    s[2] = (xy*yz - xz*yy)*r;
    s[3] = (xx*zz - xz*xz)*r;
    s[4] = (xy*xz - xx*yz)*r;
    s[5] = (xx*yy - xy*xy)*r;
    return true;
}

} // namespace gc
