// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <cmath>
#include <sstream>
#include <string>

#include "Vec3.h"

namespace spatula { namespace data {
Vec3::Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) { }

Vec3::Vec3(const double* point) : x(point[0]), y(point[1]), z(point[2]) { }

Vec3::Vec3() : x(0.0), y(0.0), z(0.0) { }

double Vec3::dot(const Vec3& b) const
{
    return x * b.x + y * b.y + z * b.z;
}

double Vec3::norm() const
{
    return std::sqrt(x * x + y * y + z * z);
}

void Vec3::normalize()
{
    const auto n = norm();
    if (n == 0) {
        return;
    }
    const double inv_norm = 1 / n;
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
}

Vec3 Vec3::cross(const Vec3& a) const
{
    return Vec3(y * a.z - z * a.y, z * a.x - x * a.z, x * a.y - y * a.x);
}

double& Vec3::operator[](const size_t i)
{
    if (i == 0) {
        return x;
    }
    if (i == 1) {
        return y;
    }
    return z;
}

const double& Vec3::operator[](size_t i) const
{
    if (i == 0) {
        return x;
    }
    if (i == 1) {
        return y;
    }
    return z;
}

template<typename number_type> Vec3 operator+(const Vec3& a, const number_type& b)
{
    return Vec3(a.x + b, a.y + b, a.z + b);
}

template<typename number_type> Vec3 operator-(const Vec3& a, const number_type& b)
{
    return Vec3(a.x - b, a.y - b, a.z - b);
}

template<typename number_type> Vec3 operator*(const Vec3& a, const number_type& b)
{
    return Vec3(a.x * b, a.y * b, a.z * b);
}

template<typename number_type> Vec3 operator/(const Vec3& a, const number_type& b)
{
    return Vec3(a.x / b, a.y / b, a.z / b);
}

template<> Vec3 operator+(const Vec3& a, const Vec3& b)
{
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

template<> Vec3 operator-(const Vec3& a, const Vec3& b)
{
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

template<> Vec3 operator*(const Vec3& a, const Vec3& b)
{
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

template<> Vec3 operator/(const Vec3& a, const Vec3& b)
{
    return Vec3(a.x / b.x, a.y / b.y, a.z / b.z);
}

template<typename number_type> Vec3& operator+=(Vec3& a, const number_type& b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    return a;
}

template<typename number_type> Vec3& operator-=(Vec3& a, const number_type& b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    return a;
}

template<typename number_type> Vec3& operator*=(Vec3& a, const number_type& b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

template<typename number_type> Vec3& operator/=(Vec3& a, const number_type& b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    return a;
}

template<> Vec3& operator+=(Vec3& a, const Vec3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

template<> Vec3& operator-=(Vec3& a, const Vec3& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

template<> Vec3& operator*=(Vec3& a, const Vec3& b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}

template<> Vec3& operator/=(Vec3& a, const Vec3& b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    return a;
}

template Vec3 operator+(const Vec3& a, const double& b);
template Vec3 operator-(const Vec3& a, const double& b);
template Vec3 operator*(const Vec3& a, const double& b);
template Vec3 operator/(const Vec3& a, const double& b);
template Vec3& operator+=(Vec3& a, const double& b);
template Vec3& operator-=(Vec3& a, const double& b);
template Vec3& operator*=(Vec3& a, const double& b);
template Vec3& operator/=(Vec3& a, const double& b);

bool operator==(const Vec3& a, const Vec3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}
}} // namespace spatula::data
