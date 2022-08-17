#include <cmath>

#include "QlmEval.h"

QlmEval::QlmEval(unsigned int m,
                 const py::array_t<double> positions,
                 const py::array_t<double> weights,
                 const py::array_t<std::complex<double>> ylms)
    : m_positions(), m_weighted_ylms()
{
    const auto unchecked_ylms = ylms.unchecked<2>();
    m_weighted_ylms.reserve(unchecked_ylms.size());
    const auto u_weights = static_cast<const double*>(weights.data());
    const double normalization = 1.0 / (4.0 * static_cast<double>(m));
    for (size_t lm {0}; lm < unchecked_ylms.shape(0); ++lm) {
        for (size_t i {0}; i < unchecked_ylms.shape(1); ++i) {
            m_weighted_ylms.push_back(normalization * u_weights[i] * unchecked_ylms(lm, i));
        }
    }
    const auto u_positions = positions.unchecked<2>();
    m_positions.assign(u_positions.data(0, 0), u_positions.data(0, 0) + u_positions.size());
}

template<typename distribution>
py::array_t<std::complex<double>> QlmEval::eval(std::shared_ptr<BondOrder<distribution>> bod)
{
    size_t n_points = m_positions.size() / 3;
    size_t n_lm = m_weighted_ylms.size() / n_points;
    py::array_t<std::complex<double>> qlms(n_lm);
    auto u_qlms = static_cast<std::complex<double>*>(qlms.mutable_data());
    const auto B_quad = bod->operator()(m_positions);
    size_t ylm_index = 0;
    for (size_t lm {0}; lm < n_lm; ++lm) {
        std::complex<double> qlm_sum = 0;
        for (size_t i {0}; i < n_points; ++i) {
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
                      const py::array_t<std::complex<double>>>())
        .def("uniform_eval", &QlmEval::eval<UniformDistribution>)
        .def("fisher_eval", &QlmEval::eval<FisherDistribution>);
}
