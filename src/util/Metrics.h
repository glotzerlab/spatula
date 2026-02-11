// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include "../data/Vec3.h"
#include <complex>
#include <numeric>
#include <vector>

namespace spatula { namespace util {
/**
 * @brief compute the Pearson correlation between two spherical harmonic expansions.
 *
 * The implementation uses some tricks to make the computation as efficient as possible compared to
 * a standard corrlation computation.
 *
 * @param f The coefficents for the first spherical harmonic expansion
 * @param g The coefficents for the second spherical harmonic expansion
 * @returns A vector of the Pearson correlation for the two expansions
 */
inline double covariance(const std::vector<std::complex<double>>& f,
                         const std::vector<std::complex<double>>& g)
{
    // For the covariance we must skip the first element as it adds a spurious
    // detection of symmetry/covariance.
    double f_cov = 0;
    double g_covar = 0;
    double mixed_covar = 0;
    for (size_t j {1}; j < f.size(); ++j) {
        f_cov += std::norm(f[j]);
        g_covar += std::norm(g[j]);
        mixed_covar += std::real(f[j] * std::conj(g[j]));
    }
    if (f_cov == 0 || g_covar == 0) {
        return 0;
    }
    return mixed_covar / std::sqrt(g_covar * f_cov);
}

template<typename T>
inline double compute_Bhattacharyya_coefficient_gaussian(const data::Vec3<T>& position,
                                                         const data::Vec3<T>& symmetrized_position,
                                                         double sigma,
                                                         double sigma_symmetrized)
{
    // 1. compute the distance between the two vectors (symmetrized_position
    //    and positions[m])
    auto r_pos = symmetrized_position - position;
    auto sigmas_squared_summed = sigma * sigma + sigma_symmetrized * sigma_symmetrized;
    // 2. compute the gaussian overlap between the two points. Bhattacharyya coefficient
    //    is used.
    double lead_term = (2 * sigma * sigma_symmetrized / sigmas_squared_summed);
    return lead_term * std::sqrt(lead_term)
           * std::exp(-static_cast<double>(r_pos.dot(r_pos)) / (4 * sigmas_squared_summed));
}
template<typename T>
inline double
compute_log_m_Bhattacharyya_coefficient_gaussian(const data::Vec3<T>& position,
                                                 const data::Vec3<T>& symmetrized_position,
                                                 double sigma,
                                                 double sigma_symmetrized)
{
    // 1. compute the distance between the two vectors (symmetrized_position
    //    and positions[m])
    auto r_pos = symmetrized_position - position;
    // Reduced equation when sigma == sigma_symmetrized
    return static_cast<double>(r_pos.dot(r_pos)) / (8.0 * (sigma * sigma_symmetrized));
}

template<typename T>
inline double compute_Bhattacharyya_coefficient_fisher(const data::Vec3<T>& position,
                                                       const data::Vec3<T>& symmetrized_position,
                                                       double kappa,
                                                       double kappa_symmetrized)
{
    auto position_norm = std::sqrt(static_cast<double>(position.dot(position)));
    auto symmetrized_position_norm = std::sqrt(static_cast<double>(symmetrized_position.dot(symmetrized_position)));
    // If position norm is zero vector means this point is at origin and contributes 1
    // to the overlap, check that with a small epsilon.
    if ((position_norm < 1e-10) && (symmetrized_position_norm < 1e-10)) {
        return 1;
    } else if ((position_norm < 1e-10) || (symmetrized_position_norm < 1e-10)) {
        return 0;
    }
    auto k1_sq = kappa * kappa;
    auto k2_sq = kappa_symmetrized * kappa_symmetrized;
    auto k1k2 = kappa * kappa_symmetrized;
    auto proj = static_cast<double>(position.dot(symmetrized_position)) / (position_norm * symmetrized_position_norm);
    return 2 * std::sqrt(k1k2 / (std::sinh(kappa) * std::sinh(kappa_symmetrized)))
           * std::sinh((std::sqrt(k1_sq + k2_sq + 2 * k1k2 * proj)) / 2)
           / std::sqrt(k1_sq + k2_sq + 2 * k1k2 * proj);
}

template<typename T>
inline double
compute_Bhattacharyya_coefficient_fisher_normalized(const data::Vec3<T>& position,
                                                    const data::Vec3<T>& symmetrized_position,
                                                    double kappa,
                                                    double kappa_symmetrized)
{
    // If position norm is zero vector means this point is at origin and contributes 1
    // to the overlap, check that with a small epsilon.
    auto k1_sq = kappa * kappa;
    auto k2_sq = kappa_symmetrized * kappa_symmetrized;
    auto k1k2 = kappa * kappa_symmetrized;
    auto proj = static_cast<double>(position.dot(symmetrized_position));
    return 2 * std::sqrt(k1k2 / (std::sinh(kappa) * std::sinh(kappa_symmetrized)))
           * std::sinh((std::sqrt(k1_sq + k2_sq + 2 * k1k2 * proj)) / 2)
           / std::sqrt(k1_sq + k2_sq + 2 * k1k2 * proj);
}
}} // namespace spatula::util
