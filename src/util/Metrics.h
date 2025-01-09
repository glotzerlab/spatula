#pragma once

#include <complex>
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
double covariance(const std::vector<std::complex<double>>& f,
                  const std::vector<std::complex<double>>& g);
}} // namespace spatula::util
