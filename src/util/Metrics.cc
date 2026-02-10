// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <complex>
#include <numeric>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Metrics.h"

namespace spatula { namespace util {
double covariance(const std::vector<std::complex<double>>& f,
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
