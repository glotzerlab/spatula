// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <cmath>

namespace spatula { namespace data {

/**
 * @brief Vec3 represents a point in 3d space and provides arithmetic operators for easy
 * manipulation. Some other functions are provided such as Vec3::dot for other common use cases.
 */
struct Vec3 {
    /// x coordinate
    double x;
    /// y coordinate
    double y;
    /// z coordinate
    double z;

    /**
     * @brief Construct a Vec3 from given Cartesian coordinates.
     */
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) { }

    /**
     * @brief Construct a Vec3 from a pointer to an array of at least length 3.
     */
    Vec3(const double* point) : x(point[0]), y(point[1]), z(point[2]) { }

    /// Construct a point at the origin.
    Vec3() : x(0.0), y(0.0), z(0.0) { }

    /**
     * @brief Compute the dot product of a dot b.
     *
     * @param b the point to compute the dot product of.
     */
    inline double dot(const Vec3& b) const
    {
        return x * b.x + y * b.y + z * b.z;
    }

    inline double norm() const
    {
        return std::sqrt(x * x + y * y + z * z);
    }

    inline void normalize()
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

    inline Vec3 cross(const Vec3& a) const
    {
        return Vec3(y * a.z - z * a.y, z * a.x - x * a.z, x * a.y - y * a.x);
    }

    inline double& operator[](const size_t i)
    {
        if (i == 0) {
            return x;
        }
        if (i == 1) {
            return y;
        }
        return z;
    }

    inline const double& operator[](size_t i) const
    {
        if (i == 0) {
            return x;
        }
        if (i == 1) {
            return y;
        }
        return z;
    }
};

/// Vec3 addition.
template<typename number_type> inline Vec3 operator+(const Vec3& a, const number_type& b)
{
    return Vec3(a.x + b, a.y + b, a.z + b);
}

/// Vec3 subtraction.
template<typename number_type> inline Vec3 operator-(const Vec3& a, const number_type& b)
{
    return Vec3(a.x - b, a.y - b, a.z - b);
}

/// Vec3 multiplication.
template<typename number_type> inline Vec3 operator*(const Vec3& a, const number_type& b)
{
    return Vec3(a.x * b, a.y * b, a.z * b);
}

/// Vec3 division.
template<typename number_type> inline Vec3 operator/(const Vec3& a, const number_type& b)
{
    return Vec3(a.x / b, a.y / b, a.z / b);
}

template<> inline Vec3 operator+(const Vec3& a, const Vec3& b)
{
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

template<> inline Vec3 operator-(const Vec3& a, const Vec3& b)
{
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

template<> inline Vec3 operator*(const Vec3& a, const Vec3& b)
{
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

template<> inline Vec3 operator/(const Vec3& a, const Vec3& b)
{
    return Vec3(a.x / b.x, a.y / b.y, a.z / b.z);
}

/// Vec3 inplace addition.
template<typename number_type> inline Vec3& operator+=(Vec3& a, const number_type& b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    return a;
}

/// Vec3 inplace subtraction.
template<typename number_type> inline Vec3& operator-=(Vec3& a, const number_type& b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    return a;
}

/// Vec3 inplace multiplication.
template<typename number_type> inline Vec3& operator*=(Vec3& a, const number_type& b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

/// Vec3 inplace division.
template<typename number_type> inline Vec3& operator/=(Vec3& a, const number_type& b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    return a;
}

template<> inline Vec3& operator+=(Vec3& a, const Vec3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

template<> inline Vec3& operator-=(Vec3& a, const Vec3& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

template<> inline Vec3& operator*=(Vec3& a, const Vec3& b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}

template<> inline Vec3& operator/=(Vec3& a, const Vec3& b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    return a;
}

/// Vec3 equality
inline bool operator==(const Vec3& a, const Vec3& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

}} // namespace spatula::data
