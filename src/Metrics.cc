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

WeightedPNormBase::WeightedPNormBase() : m_weights(), m_normalization(0.0) { }

WeightedPNormBase::WeightedPNormBase(const std::vector<double>& weights)
    : m_weights(weights), m_normalization(std::accumulate(m_weights.begin(), m_weights.end(), 0.0))
{
}

template<unsigned int p> WeightedPNorm<p>::WeightedPNorm() : WeightedPNormBase() { }

template<unsigned int p>
WeightedPNorm<p>::WeightedPNorm(const std::vector<double>& weights) : WeightedPNormBase(weights)
{
}

template<unsigned int p>
double WeightedPNorm<p>::operator()(const std::vector<double>& vector) const
{
    double metric = 0;
    if (m_weights.size() != 0) {
        for (size_t i {0}; i < vector.size(); ++i) {
            double v = vector[i];
            if constexpr (p % 2 == 0) {
                v = std::abs(v);
            }
            metric += m_weights[i] * std::pow(v, p);
        }
        metric /= m_normalization;
    } else {
        for (size_t i {0}; i < vector.size(); ++i) {
            double v = vector[i];
            if constexpr (p % 2 == 0) {
                v = std::abs(v);
            }
            metric += std::pow(v, p);
        }
    }
    return std::pow(metric, 1 / static_cast<double>(p));
}

template class WeightedPNorm<1>;
template class WeightedPNorm<2>;
template class WeightedPNorm<3>;
template class WeightedPNorm<4>;
