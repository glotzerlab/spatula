// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <cmath>
#include <complex>
#include <iterator>
#include <utility>
#include <vector>

#include "../data/RotationMatrix.h"
#include "../data/Vec3.h"
#include "fastmath.h"

namespace spatula { namespace util {

// Bring Vec3 and Quaternion into namespace.
using namespace spatula::data;

using vec3_iter = decltype(std::declval<std::vector<Vec3>>().begin());
using cvec3_iter = decltype(std::declval<const std::vector<Vec3>>().begin());

/// Compute and return the angle (in radians) between two vectors in 3D.
inline float fast_angle_eucledian(const Vec3& ref_x, const Vec3& x)
{
    return std::acos(ref_x.dot(x));
}

/// Rotate a single point x using rotation matrix R and place the result in x_prime.
inline void single_rotate(const Vec3& x, Vec3& x_prime, const RotationMatrix& R)
{
    x_prime.x = R[0] * x.x + R[1] * x.y + R[2] * x.z;
    x_prime.y = R[3] * x.x + R[4] * x.y + R[5] * x.z;
    x_prime.z = R[6] * x.x + R[7] * x.y + R[8] * x.z;
};

/**
 * @brief Rotate an interator of points via the rotation matrix R.
 * The points rotated are given by @c (auto it = points_begin; it < points_end; ++it).
 *
 * @tparam IntputIterator An input iterator (or derived iterator concept).
 * @param points_begin constant iterator to the beginning of points to rotate.
 * @param points_end constant iterator to the end of points to rotate.
 * @param rotated_points_it iterator to the starting vector location to place rotated positions in.
 * @param R The rotation matrix given in row column order.
 */
inline void rotate_matrix(cvec3_iter points_begin,
                          cvec3_iter points_end,
                          vec3_iter rotated_points_it,
                          const RotationMatrix& R)
{
    for (auto it = points_begin; it != points_end; ++it, ++rotated_points_it) {
        single_rotate(*it, *rotated_points_it, R);
    }
}

/**
 * @brief Convert a Vec3 representing an axis, angle rotation parametrization to a rotation matrix.
 *
 * This method assumes that \f$ || v || = \theta \f$ and \f$ x = \frac{v}{||v||} \f$ where \f$ x \f$
 * is the axis of rotation.
 *
 * @param v The 3-vector to convert to a rotation matrix.
 */
inline RotationMatrix to_rotation_matrix(const Vec3& v)
{
    const auto angle = v.norm();
    if (std::abs(angle) < 1e-7) {
        return RotationMatrix {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    }
    const auto axis = v / angle;
    const float c {static_cast<float>(std::cos(angle))};
    const float s {static_cast<float>(std::sin(angle))};
    const float C = 1.0f - c;
    const auto sv = axis * s;
    return RotationMatrix {
        C * axis.x * axis.x + c,
        C * axis.x * axis.y - sv.z,
        C * axis.x * axis.z + sv.y,
        C * axis.y * axis.x + sv.z,
        C * axis.y * axis.y + c,
        C * axis.y * axis.z - sv.x,
        C * axis.z * axis.x - sv.y,
        C * axis.z * axis.y + sv.x,
        C * axis.z * axis.z + c,
    };
}

/**
 * @brief Returns a vector of Vec3 of normalized distances. Each point in distances is normalized
 * and converted to a Vec3
 *
 * @tparam T The floating point type (float or double)
 * @param distances A Raw pointer to an array of floats interpreted as Vec3.
 * @returns a vector of Vec3 that is the same size as distances with each vector in the same
 * direction but with unit magnitude.
 */
inline std::vector<Vec3> normalize_distances(const float* distances,
                                             std::pair<size_t, size_t> slice)
{
    auto normalized_distances = std::vector<Vec3>();
    normalized_distances.reserve((slice.second - slice.first) / 3);
    // In C++ 23 used strided view with a transform.
    for (size_t i = slice.first; i < slice.second; i += 3) {
        const auto point = Vec3(distances[i], distances[i + 1], distances[i + 2]);
        const float norm = std::sqrt(point.dot(point));
        if (norm == 0.0f) {
            normalized_distances.emplace_back(point);
        } else {
            normalized_distances.emplace_back(point / norm);
        }
    }
    return normalized_distances;
}

/**
 * @brief Perform a symmetrization of a spherical harmonic expansion via a Wigner D matrix.
 *
 * @param qlms The spherical harmonic expansion coefficients.
 * @param D_ij The Wigner D matrix to symmetrize Qlms by
 * @param sym_qlm_buf The buffer/array to store the symmetrized Qlms
 * @param max_l The max_l of the provided Qlms and D_ij. This is not strictly necessary, but it
 * prevents the need to determine the max_l from the qlms vector or create a custom struct that
 * stores this.
 */
inline void symmetrize_qlm(const std::vector<std::complex<double>>& qlms,
                           const std::vector<std::complex<double>>& D_ij,
                           std::vector<std::complex<double>>& sym_qlm_buf,
                           unsigned int max_l)
{
    sym_qlm_buf.clear();
    sym_qlm_buf.reserve(qlms.size());
    size_t qlm_i {0};
    size_t dij_index {0};
    for (size_t l {0}; l < max_l + 1; ++l) {
        const size_t max_m {2 * l + 1};
        for (size_t m_prime {0}; m_prime < max_m; ++m_prime) {
            std::complex<double> sym_qlm {0.0, 0.0};
            for (size_t m {0}; m < max_m; ++m) {
                sym_qlm += qlms[qlm_i + m] * D_ij[dij_index];
                ++dij_index;
            }
            sym_qlm_buf.emplace_back(sym_qlm);
        }
        qlm_i += max_m;
    }
}

}} // namespace spatula::util
