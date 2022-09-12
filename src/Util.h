#pragma once

#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "Vec3.h"

namespace py = pybind11;

/// Compute and return the angle (in radians) between two vectors in 3D.
double fast_angle_eucledian(const Vec3& ref_x, const Vec3& x);

/// Rotate a single point x using rotation matrix R and place the result in x_prime.
void single_rotate(const Vec3& x, Vec3& x_prime, const std::vector<double>& R);

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
