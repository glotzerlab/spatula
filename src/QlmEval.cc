#include <cmath>

#include "QlmEval.h"

QlmEval::QlmEval(unsigned int m,
                 const py::array_t<double> quad_theta,
                 const py::array_t<double> quad_phi,
                 const py::array_t<double> weights,
                 const py::array_t<std::complex<double>> ylms)
    : m_sin_theta(), m_cos_theta(), m_weighted_ylms(), m_phi()
{
    auto n_points = static_cast<size_t>(quad_theta.size());
    m_sin_theta.reserve(n_points);
    m_cos_theta.reserve(n_points);
    m_phi.reserve(n_points);

    const auto u_theta = static_cast<const double*>(quad_theta.data());
    const auto u_phi = static_cast<const double*>(quad_phi.data());
    for (size_t i {0}; i < n_points; ++i) {
        const auto theta = u_theta[i] - M_PI_2;
        m_sin_theta.push_back(std::sin(theta));
        m_cos_theta.push_back(std::cos(theta));
        m_phi.push_back(u_phi[i]);
    }
    const auto unchecked_ylms = ylms.unchecked<2>();
    m_weighted_ylms.reserve(n_points * unchecked_ylms.shape(1));
    const auto u_weights = static_cast<const double*>(weights.data());
    const double normalization = 1.0 / (4.0 * static_cast<double>(m));
    for (size_t lm {0}; lm < unchecked_ylms.shape(0); ++lm) {
        for (size_t i {0}; i < n_points; ++i) {
            m_weighted_ylms.push_back(normalization * u_weights[i] * unchecked_ylms(lm, i));
        }
    }
}

template<typename distribution>
py::array_t<std::complex<double>> QlmEval::eval(std::shared_ptr<BondOrder<distribution>> bod)
{
    size_t n_lm = m_weighted_ylms.size() / m_phi.size();
    py::array_t<std::complex<double>> qlms(n_lm);
    auto u_qlms = static_cast<std::complex<double>*>(qlms.mutable_data());
    ;
    const auto B_quad = bod->fast_call(m_sin_theta, m_cos_theta, m_phi);
    size_t ylm_index = 0;
    for (size_t lm {0}; lm < n_lm; ++lm) {
        std::complex<double> qlm_sum = 0;
        for (size_t i {0}; i < m_phi.size(); ++i) {
            qlm_sum += B_quad[i] * m_weighted_ylms[ylm_index];
            ++ylm_index;
        }
        u_qlms[lm] = qlm_sum;
    }
    return qlms;
}

void export_qlm_eval(py::module& m)
{
    py::class_<QlmEval>(m, "QlmEval")
        .def(py::init<unsigned int,
                      const py::array_t<double>,
                      const py::array_t<double>,
                      const py::array_t<double>,
                      const py::array_t<std::complex<double>>>())
        .def("uniform_eval", &QlmEval::eval<UniformDistribution>)
        .def("fisher_eval", &QlmEval::eval<FisherDistribution>);
}
