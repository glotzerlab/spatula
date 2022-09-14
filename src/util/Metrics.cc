#include <complex>
#include <numeric>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Metrics.h"

namespace pgop { namespace util {
double covariance(const std::vector<std::complex<double>>& qlms,
                  const std::vector<std::complex<double>>& sym_qlms)
{
    // For the covariance we must skip the first element as it adds a spurious
    // detection of symmetry/covariance.
    double qlm_cov = 0;
    double sym_covar = 0;
    double mixed_covar = 0;
    for (size_t j {1}; j < qlms.size(); ++j) {
        qlm_cov += std::norm(qlms[j]);
        sym_covar += std::norm(sym_qlms[j]);
        mixed_covar += std::real(qlms[j] * std::conj(sym_qlms[j]));
    }
    return mixed_covar / std::sqrt(sym_covar * qlm_cov);
}
}} // namespace pgop::util
