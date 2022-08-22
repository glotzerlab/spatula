#include <complex>
#include <numeric>
#include <string>

#include "Metrics.h"

py::array_t<double> covariance(py::array_t<std::complex<double>> qlms,
                               py::array_t<std::complex<double>> sym_qlms)
{
    const auto u_qlms = qlms.unchecked<1>();
    const double qlm_cov
        = std::accumulate(u_qlms.data(0),
                          u_qlms.data(0) + u_qlms.size(),
                          0.0,
                          [](const auto& x, const auto& y) { return x + std::norm(y); });
    const auto u_sym_qlms = sym_qlms.unchecked<2>();
    auto covar = py::array_t<double>(u_sym_qlms.shape(0));
    auto u_covar = static_cast<double*>(covar.mutable_data(0));
    for (size_t i {0}; i < u_sym_qlms.shape(0); ++i) {
        double sym_covar = 0;
        double mixed_covar = 0;
        for (size_t j {0}; j < u_sym_qlms.shape(1); ++j) {
            sym_covar += std::norm(u_sym_qlms(i, j));
            mixed_covar += std::real(u_qlms(j) * std::conj(u_sym_qlms(i, j)));
        }
        u_covar[i] = mixed_covar / std::sqrt(sym_covar * qlm_cov);
    }
    return covar;
}

WeightedPNormBase::WeightedPNormBase(std::vector<double>& weights)
    : m_weights(weights), m_normalization(std::accumulate(m_weights.begin(), m_weights.end(), 0.0))
{
}

template<unsigned int p>
WeightedPNorm<p>::WeightedPNorm(std::vector<double>& weights) : WeightedPNormBase(weights)
{
}

template<unsigned int p> double WeightedPNorm<p>::operator()(py::array_t<double> vector)
{
    const auto* u_vector = static_cast<const double*>(vector.data(0));
    double metric = 0;
    if (m_weights.size() != 0) {
        for (size_t i {0}; i < m_weights.size(); ++i) {
            double v = u_vector[i];
            if constexpr (p % 2 == 0) {
                v = std::abs(v);
            }
            metric += m_weights[i] * std::pow(v, p);
        }
        metric /= m_normalization;
    } else {
        for (size_t i {0}; i < vector.size(); ++i) {
            double v = u_vector[i];
            if constexpr (p % 2 == 0) {
                v = std::abs(v);
            }
            metric += std::pow(v, p);
        }
    }
    return std::pow(metric, 1 / static_cast<double>(p));
}

template<unsigned int p> void export_pnorm(py::module& m)
{
    auto name = "Weighted" + std::to_string(p) + "Norm";
    py::class_<WeightedPNorm<p>>(m, name.c_str())
        .def(py::init<std::vector<double>&>())
        .def("__call__", &WeightedPNorm<p>::operator());
}

void export_metrics(py::module& m)
{
    m.def("covariance_score", covariance);
    export_pnorm<1>(m);
    export_pnorm<2>(m);
    export_pnorm<3>(m);
}
