#pragma once

#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../data/Vec3.h"

namespace py = pybind11;

namespace pgop { namespace util {

// Bring Vec3 and Quaternion into namespace.
using namespace pgop::data;

/// Compute and return the angle (in radians) between two vectors in 3D.
double fast_angle_eucledian(const Vec3& ref_x, const Vec3& x);

/// Rotate a single point x using rotation matrix R and place the result in x_prime.
void single_rotate(const Vec3& x, Vec3& x_prime, const std::vector<double>& R);

/**
 * @brief Rotate an interator of points via the rotation matrix R.
 * The points rotated are given by @c (auto it = points_begin; it < points_end; ++it).
 *
 * @param points_begin constant iterator to the beginning of points to rotate.
 * @param points_end constant iterator to the end of points to rotate.
 * @param rotated_points_it iterator to the starting vector location to place rotated positions in.
 * @param R The rotation matrix given in row column order.
 */
void rotate_matrix(std::vector<Vec3>::const_iterator points_begin,
                   std::vector<Vec3>::const_iterator points_end,
                   std::vector<Vec3>::iterator rotated_points_it,
                   const std::vector<double>& R);

/**
 * @brief Rotate a set of points using a rotation matrix. Uses the Euler angle convention XYZ
 * intrinsic convention @todo check to see if this is correct. The points rotated are given by @c
 * (auto it = points_begin; it < points_end; ++it).
 *
 * @param points_begin constant iterator to the beginning of points to rotate.
 * @param points_end constant iterator to the end of points to rotate.
 * @param rotated_points_it iterator to the starting vector location to place rotated positions in.
 * @param alpha Euler angle
 * @param beta Euler angle
 * @param gamma Euler angle
 */
void rotate_euler(const std::vector<Vec3>::const_iterator points_begin,
                  const std::vector<Vec3>::const_iterator points_end,
                  std::vector<Vec3>::iterator rotated_points_it,
                  double alpha,
                  double beta,
                  double gamma);

/**
 * @brief Rotate a set of points using a rotation matrix. Uses the Euler angle convention XYZ
 * intrinsic convention @todo check to see if this is correct.
 *
 * @param points_begin constant iterator to the beginning of points to rotate.
 * @param points_end constant iterator to the end of points to rotate.
 * @param rotated_points_it iterator to the starting vector location to place rotated positions in.
 * @param rotation a size 9 vector that is a 3x3 rotation matrix. This is the matrix used to rotate
 * the points given by @c (auto it = points_begin; it < points_end; ++it).
 */
void rotate_euler(std::vector<Vec3>::const_iterator points_begin,
                  std::vector<Vec3>::const_iterator points_end,
                  std::vector<Vec3>::iterator rotated_points_it,
                  const std::vector<double>& rotation);

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
std::vector<Vec3> normalize_distances(const py::array_t<double> distances);

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
}} // namespace pgop::util
