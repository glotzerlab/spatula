#include <complex>
#include <numeric>

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

void export_metrics(py::module& m)
{
    m.def("covariance_score", covariance);
}
