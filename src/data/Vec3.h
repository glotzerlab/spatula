// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <cmath>

namespace spatula { namespace data {

/**
 * @brief Vec3 represents a point in 3d space and provides arithmetic operators for easy
 * manipulation. Some other functions are provided such as Vec3::dot for other common use cases.
 *
 * @tparam T The floating point type (float or double)
 */
template<typename T> struct Vec3 {
    /// x coordinate
    T x;
    /// y coordinate
    T y;
    /// z coordinate
    T z;

    /**
     * @brief Construct a Vec3 from given Cartesian coordinates.
     */
    Vec3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) { }

    /**
     * @brief Construct a Vec3 from a pointer to an array of at least length 3.
     */
    explicit Vec3(const T* point) : x(point[0]), y(point[1]), z(point[2]) { }

    /// Construct a point at the origin.
    Vec3() : x(T(0)), y(T(0)), z(T(0)) { }

    /**
     * @brief Compute the dot product of a dot b.
     *
     * @param b the point to compute the dot product of.
     */
    inline T dot(const Vec3& b) const
    {
        return x * b.x + y * b.y + z * b.z;
    }

    inline T norm() const
    {
        return std::sqrt(x * x + y * y + z * z);
    }

    inline void normalize()
    {
        const auto n = norm();
        if (n == T(0)) {
            return;
        }
        const T inv_norm = T(1) / n;
        x *= inv_norm;
        y *= inv_norm;
        z *= inv_norm;
    }

    inline Vec3 cross(const Vec3& a) const
    {
        return Vec3(y * a.z - z * a.y, z * a.x - x * a.z, x * a.y - y * a.x);
    }

    inline T& operator[](const size_t i)
    {
        if (i == 0) {
            return x;
        }
        if (i == 1) {
            return y;
        }
        return z;
    }

    inline const T& operator[](size_t i) const
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
template<typename T, typename number_type>
inline Vec3<T> operator+(const Vec3<T>& a, const number_type& b)
{
    return Vec3<T>(a.x + b, a.y + b, a.z + b);
}

/// Vec3 subtraction.
template<typename T, typename number_type>
inline Vec3<T> operator-(const Vec3<T>& a, const number_type& b)
{
    return Vec3<T>(a.x - b, a.y - b, a.z - b);
}

/// Vec3 multiplication.
template<typename T, typename number_type>
inline Vec3<T> operator*(const Vec3<T>& a, const number_type& b)
{
    return Vec3<T>(a.x * b, a.y * b, a.z * b);
}

/// Vec3 division.
template<typename T, typename number_type>
inline Vec3<T> operator/(const Vec3<T>& a, const number_type& b)
{
    return Vec3<T>(a.x / b, a.y / b, a.z / b);
}

template<typename T> inline Vec3<T> operator+(const Vec3<T>& a, const Vec3<T>& b)
{
    return Vec3<T>(a.x + b.x, a.y + b.y, a.z + b.z);
}

template<typename T> inline Vec3<T> operator-(const Vec3<T>& a, const Vec3<T>& b)
{
    return Vec3<T>(a.x - b.x, a.y - b.y, a.z - b.z);
}

template<typename T> inline Vec3<T> operator*(const Vec3<T>& a, const Vec3<T>& b)
{
    return Vec3<T>(a.x * b.x, a.y * b.y, a.z * b.z);
}

template<typename T> inline Vec3<T> operator/(const Vec3<T>& a, const Vec3<T>& b)
{
    return Vec3<T>(a.x / b.x, a.y / b.y, a.z / b.z);
}

/// Vec3 inplace addition.
template<typename T, typename number_type>
inline Vec3<T>& operator+=(Vec3<T>& a, const number_type& b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    return a;
}

/// Vec3 inplace subtraction.
template<typename T, typename number_type>
inline Vec3<T>& operator-=(Vec3<T>& a, const number_type& b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    return a;
}

/// Vec3 inplace multiplication.
template<typename T, typename number_type>
inline Vec3<T>& operator*=(Vec3<T>& a, const number_type& b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

/// Vec3 inplace division.
template<typename T, typename number_type>
inline Vec3<T>& operator/=(Vec3<T>& a, const number_type& b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    return a;
}

template<typename T> inline Vec3<T>& operator+=(Vec3<T>& a, const Vec3<T>& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

template<typename T> inline Vec3<T>& operator-=(Vec3<T>& a, const Vec3<T>& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

template<typename T> inline Vec3<T>& operator*=(Vec3<T>& a, const Vec3<T>& b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}

template<typename T> inline Vec3<T>& operator/=(Vec3<T>& a, const Vec3<T>& b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    return a;
}

/// Vec3 equality
template<typename T> inline bool operator==(const Vec3<T>& a, const Vec3<T>& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

// Typedefs for common precision types
using Vec3d = Vec3<double>;
using Vec3f = Vec3<float>;

}} // namespace spatula::data
