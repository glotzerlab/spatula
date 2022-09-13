#include <complex>
#include <numeric>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Metrics.h"

std::vector<double> covariance(const std::vector<std::complex<double>>& qlms,
                               const std::vector<std::vector<std::complex<double>>>& sym_qlms)
{
    // For the covariance we must skip the first element as it adds a spurious
    // detection of symmetry/covariance.
    const double qlm_cov
        = std::accumulate(qlms.begin() + 1, qlms.end(), 0.0, [](const auto& sum, const auto& y) {
              return std::norm(y) + sum;
          });
    auto covar = std::vector<double>();
    const size_t N_sym = sym_qlms.size();
    covar.reserve(N_sym);
    for (size_t i {0}; i < N_sym; ++i) {
        const auto& sym_i_qlms = sym_qlms[i];
        double sym_covar = 0;
        double mixed_covar = 0;
        for (size_t j {1}; j < qlms.size(); ++j) {
            sym_covar += std::norm(sym_i_qlms[j]);
            mixed_covar += std::real(qlms[j] * std::conj(sym_i_qlms[j]));
        }
        const auto K = mixed_covar / std::sqrt(sym_covar * qlm_cov);
        covar.emplace_back(K);
    }
    return covar;
}

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
