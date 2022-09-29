#pragma once

#include <cmath>
#include <complex>
#include <concepts>
#include <iterator>
#include <ranges>
#include <vector>

#include "../data/Vec3.h"

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
 * This method is templated to enable more easy refactoring of container types in PGOP.cc.
 *
 * @tparam IntputIterator An input iterator (or derived iterator concept).
 * @tparam OutputIterator An output iterator (or derived iterator concept).
 *
 * @param points_begin constant iterator to the beginning of points to rotate.
 * @param points_end constant iterator to the end of points to rotate.
 * @param rotated_points_it iterator to the starting vector location to place rotated positions in.
 * @param R The rotation matrix given in row column order.
 */
template<std::input_iterator InputIterator, std::output_iterator<Vec3> OutputIterator>
void rotate_matrix(const InputIterator points_begin,
                   const InputIterator points_end,
                   OutputIterator rotated_points_it,
                   const std::vector<double>& R)
{
    for (auto it = points_begin; it != points_end; ++it, ++rotated_points_it) {
        single_rotate(*it, *rotated_points_it, R);
    }
}

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
template<std::ranges::input_range range_type>
requires std::floating_point<std::ranges::range_value_t<range_type>> std::vector<Vec3>
normalize_distances(const range_type& distances)
{
    auto normalized_distances = std::vector<Vec3>();
    normalized_distances.reserve(distances.size() / 3);
    // In C++ 23 used strided view with a transform.
    for (auto it = distances.begin(); it < distances.end(); it += 3) {
        const auto point = Vec3(it[0], it[1], it[2]);
        const double norm = 1 / std::sqrt(point.dot(point));
        normalized_distances.emplace_back(point * norm);
    }
    return normalized_distances;
}

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
