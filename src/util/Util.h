// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <cmath>
#include <complex>
#include <iterator>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../data/Vec3.h"

namespace py = pybind11;

namespace spatula { namespace util {

// Bring Vec3 and Quaternion into namespace.
using namespace spatula::data;

using vec3_iter = decltype(std::declval<std::vector<Vec3>>().begin());
using cvec3_iter = decltype(std::declval<const std::vector<Vec3>>().begin());

/// Compute and return the angle (in radians) between two vectors in 3D.
double fast_angle_eucledian(const Vec3& ref_x, const Vec3& x);

/// Rotate a single point x using rotation matrix R and place the result in x_prime.
void single_rotate(const Vec3& x, Vec3& x_prime, const std::vector<double>& R);

/**
 * @brief Rotate an interator of points via the rotation matrix R.
 * The points rotated are given by @c (auto it = points_begin; it < points_end; ++it).
 *
 * This method is templated to enable more easy refactoring of container types in PGOP.cc.
 *
 * @tparam IntputIterator An input iterator (or derived iterator concept).
 * @param points_begin constant iterator to the beginning of points to rotate.
 * @param points_end constant iterator to the end of points to rotate.
 * @param rotated_points_it iterator to the starting vector location to place rotated positions in.
 * @param R The rotation matrix given in row column order.
 */
void rotate_matrix(cvec3_iter points_begin,
                   cvec3_iter points_end,
                   vec3_iter rotated_points_it,
                   const std::vector<double>& R);

/**
 * @brief Convert a Vec3 representing an axis, angle rotation parametrization to a rotation matrix.
 *
 * This method assumes that \f$ || v || = \theta \f$ and \f$ x = \frac{v}{||v||} \f$ where \f$ x \f$
 * is the axis of rotation.
 *
 * @param v the rotation coded according to the 3 vector axis angle parametrization.
 */
std::vector<double> to_rotation_matrix(const Vec3& v);

/**
 * @brief Compute the rotation matrix for the given Euler angles in ??? convention.
 *
 * @param alpha Euler angle
 * @param beta Euler angle
 * @param gamma Euler angle
 * @returns the rotation matrix as a 1d vector.
 */
std::vector<double> compute_rotation_matrix(double alpha, double beta, double gamma);

/**
 * @brief Compute the rotation matrix for the given Euler angles provided by a vector in ???
 * convention.
 *
 * @param rotation a 3 sized vector which contains Euler angles.
 * @returns the rotation matrix as a 1d vector.
 */
std::vector<double> compute_rotation_matrix(const std::vector<double>& rotation);

/**
 * @brief Returns a vector of Vec3 of normalized distances. Each point in distances is normalized
 * and converted to a Vec3
 *
 * @param distances a NumPy array wrapped by Pybind11 of points in 3D space.
 * @returns a vector of Vec3 that is the same size as distances with each vector in the same
 * direction but with unit magnitude.
 */
std::vector<Vec3> normalize_distances(const double* distances, std::pair<size_t, size_t> slice);

/**
 * @brief Return a vector of linearly spaced points between start and end.
 *
 * @param start The starting value.
 * @param end The final or n-th + 1 value according to the value of include_end
 * @param n The number of points in the vector.
 * @param include_end Whether the last point is at or before @p end.
 */
std::vector<double> linspace(double start, double end, unsigned int n, bool include_end = true);

/**
 * @brief Given a WignerD matrix and spherical harmonic expansion coefficients \f$ Q_{m}^{l} \f$
 * compute the symmetrized expansion's coefficients.
 *
 * For reasons of performance this uses an existing vector's memory buffer to avoid memory
 * allocations.
 *
 * @param qlms The spherical harmonic expansion coefficients.
 * @param D_ij The WignerD matrix for a given symmetry or point group.
 * @param sym_qlm_buf The vector to place the symmetrized expansion coefficients into. For best
 * performance the capacity should be the size of qlms.
 * @param max_l The maximum \f$ l \f$ present in @p D_ij and @p qlms.
 */
void symmetrize_qlm(const std::vector<std::complex<double>>& qlms,
                    const std::vector<std::complex<double>>& D_ij,
                    std::vector<std::complex<double>>& sym_qlm_buf,
                    unsigned int max_l);

void export_util(py::module& m);
}} // namespace spatula::util
