// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <cmath>
#include <iterator>
#include <numeric>
#include <pybind11/stl.h>

#include "Util.h"

/**
 * @brief Utility methods primarily for rotating vectors.
 *
 * Many methods extensively use references to avoid allocatings and copying. This does result in
 * function signatures reminescent of C, but is necessary the best possible performance.
 */

namespace spatula { namespace util {
/**
 * @brief Get the angle between two vectors.
 *
 * Assumes points are on the unit sphere
 *
 * @param ref_x the vector to measure an angle from
 * @param x the vector to measure the angle from ref_x
 */
double fast_angle_eucledian(const Vec3& ref_x, const Vec3& x)
{
    return std::acos(ref_x.dot(x));
}

/**
 * @brief Convert from the 3-vector representation to a rotation matrix.
 *
 * The 3-vector has a rotation axis of its unit vector and angle in radians equal to its norm.
 *
 * Rotating via a rotation matrix is more efficient operation than either direct quaternion or
 * axis-angle rotations. Already by rotating two point the same converting first is more efficient.
 *
 *
 * @param v The 3-vector to convert to a rotation matrix.
 */
std::vector<double> to_rotation_matrix(const Vec3& v)
{
    const auto angle = v.norm();
    if (std::abs(angle) < 1e-7) {
        return std::vector<double> {{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}};
    }
    const auto axis = v / angle;
    const double c {std::cos(angle)}, s {std::sin(angle)};
    const double C = 1 - c;
    const auto sv = axis * s;
    return std::vector<double> {{
        C * axis.x * axis.x + c,
        C * axis.x * axis.y - sv.z,
        C * axis.x * axis.z + sv.y,
        C * axis.y * axis.x + sv.z,
        C * axis.y * axis.y + c,
        C * axis.y * axis.z - sv.x,
        C * axis.z * axis.x - sv.y,
        C * axis.z * axis.y + sv.x,
        C * axis.z * axis.z + c,
    }};
}

/**
 * @brief Rotate a single vector via a provided rotation matrix.
 *
 * @param x The vector to rotate.
 * @param x_prime The location to store the rotated vector.
 */
void single_rotate(const Vec3& x, Vec3& x_prime, const std::vector<double>& R)
{
    x_prime.x = R[0] * x.x + R[1] * x.y + R[2] * x.z;
    x_prime.y = R[3] * x.x + R[4] * x.y + R[5] * x.z;
    x_prime.z = R[6] * x.x + R[7] * x.y + R[8] * x.z;
};

/**
 * @brief rotate an array of vectors by a provided rotation matrix.
 *
 * @param points_begin The start of the vectors to rotate iterator.
 * @param points_end The end of the vectors to rotate iterator.
 * @param rotated_points_it The start of the location to store rotated vectors.
 * @param R The rotation matrix to rotate vectors by.
 */
void rotate_matrix(cvec3_iter points_begin,
                   cvec3_iter points_end,
                   vec3_iter rotated_points_it,
                   const std::vector<double>& R)
{
    for (auto it = points_begin; it != points_end; ++it, ++rotated_points_it) {
        single_rotate(*it, *rotated_points_it, R);
    }
}

/**
 * @brief Normalize distance vectors to unit vectors.
 *
 * @param distances The array of distances to normalize. Note that the array is single dimensional
 * and the second dimension is implicit in the ordering. This impacts indexing into the array.
 * @param slice the index to start and end the normalization in absolute indices.
 */
std::vector<Vec3> normalize_distances(const double* distances, std::pair<size_t, size_t> slice)
{
    auto normalized_distances = std::vector<Vec3>();
    normalized_distances.reserve((slice.second - slice.first) / 3);
    // In C++ 23 used strided view with a transform.
    for (size_t i = slice.first; i < slice.second; i += 3) {
        const auto point = Vec3(distances[i], distances[i + 1], distances[i + 2]);
        const double norm = std::sqrt(point.dot(point));
        if (norm == 0) {
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
void symmetrize_qlm(const std::vector<std::complex<double>>& qlms,
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

void export_util(py::module& m)
{
    m.def("to_rotation_matrix", &to_rotation_matrix);
    m.def("single_rotate", &single_rotate);
}
}} // namespace spatula::util
