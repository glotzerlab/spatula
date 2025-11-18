// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <complex>
#include <numeric> // Added for std::sqrt
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
}} // namespace spatula::util
