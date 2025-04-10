// Copyright (c) 2010-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>

#pragma once

namespace spatula { namespace data {

namespace py = pybind11;

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
    Vec3(double x, double y, double z);

    /**
     * @brief Construct a Vec3 from a pointer to an array of at least length 3.
     */
    Vec3(const double* point);

    /// Construct a point at the origin.
    Vec3();

    /**
     * @brief Compute the dot product of a dot b.
     *
     * @param b the point to compute the dot product of.
     */
    double dot(const Vec3& b) const;

    double norm() const;

    void normalize();

    Vec3 cross(const Vec3& a) const;

    double& operator[](size_t i);
    const double& operator[](size_t i) const;
};

/// Vec3 addition.
template<typename number_type> Vec3 operator+(const Vec3& a, const number_type& b);

/// Vec3 subtraction.
template<typename number_type> Vec3 operator-(const Vec3& a, const number_type& b);

/// Vec3 multiplication.
template<typename number_type> Vec3 operator*(const Vec3& a, const number_type& b);

/// Vec3 division.
template<typename number_type> Vec3 operator/(const Vec3& a, const number_type& b);

/// Vec3 inplace addition.
template<typename number_type> Vec3& operator+=(Vec3& a, const number_type& b);

/// Vec3 inplace subtraction.
template<typename number_type> Vec3& operator-=(Vec3& a, const number_type& b);

/// Vec3 inplace multiplication.
template<typename number_type> Vec3& operator*=(Vec3& a, const number_type& b);

/// Vec3 inplace division.
template<typename number_type> Vec3& operator/=(Vec3& a, const number_type& b);

/// Vec3 equality
bool operator==(const Vec3& a, const Vec3& b);

void export_Vec3(py::module& m);
}} // namespace spatula::data
