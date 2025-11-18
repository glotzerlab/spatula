// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include "Vec3.h"
#include <pybind11/operators.h> // Needed for py::self in export-vec.cc, but also good to have here for consistency

namespace spatula { namespace data {

template Vec3 operator+(const Vec3& a, const double& b);
template Vec3 operator-(const Vec3& a, const double& b);
template Vec3 operator*(const Vec3& a, const double& b);
template Vec3 operator/(const Vec3& a, const double& b);
template Vec3& operator+=(Vec3& a, const double& b);
template Vec3& operator-=(Vec3& a, const double& b);
template Vec3& operator*=(Vec3& a, const double& b);
template Vec3& operator/=(Vec3& a, const double& b);

}} // namespace spatula::data
