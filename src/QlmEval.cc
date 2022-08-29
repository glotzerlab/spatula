#include <algorithm>
#include <cmath>
#include <iterator>

#include "QlmEval.h"

QlmEval::QlmEval(unsigned int m,
                 const py::array_t<double> positions,
                 const py::array_t<double> weights,
                 const py::array_t<std::complex<double>> ylms)
    : m_n_points(ylms.shape(1)), m_n_lms(ylms.shape(0)), m_positions(), m_weighted_ylms()
{
    m_weighted_ylms.reserve(m_n_lms);
    const auto unchecked_ylms = ylms.unchecked<2>();
    const auto u_weights = static_cast<const double*>(weights.data());
    const double normalization = 1.0 / (4.0 * static_cast<double>(m));
    for (size_t lm {0}; lm < m_n_lms; ++lm) {
        auto ylm = std::vector<std::complex<double>>();
        ylm.reserve(m_n_points);
        for (size_t i {0}; i < m_n_points; ++i) {
            ylm.emplace_back(normalization * u_weights[i] * unchecked_ylms(lm, i));
        }
        m_weighted_ylms.emplace_back(ylm);
    }
    const auto u_positions = positions.unchecked<2>();
    m_positions.reserve(positions.shape(0));
    for (size_t i {0}; i < positions.shape(0); ++i) {
        m_positions.emplace_back(u_positions.data(i, 0));
    }
}

template<typename distribution_type>
std::vector<std::complex<double>> QlmEval::eval(const BondOrder<distribution_type>& bod) const
{
    std::vector<std::complex<double>> qlms;
    qlms.reserve(m_n_lms);
    const auto B_quad = bod(m_positions);
    std::transform(m_weighted_ylms.begin(),
                   m_weighted_ylms.end(),
                   std::back_insert_iterator(qlms),
                   [&B_quad](const auto& w_ylm) {
                       std::complex<double> dot = 0;
                       for (size_t i {0}; i < w_ylm.size(); ++i) {
                           dot += B_quad[i] * w_ylm[i];
                       }
                       return dot;
                   });
    return qlms;
}

unsigned int QlmEval::getNlm() const
{
    return m_n_lms;
};

template std::vector<std::complex<double>>
QlmEval::eval<UniformDistribution>(const BondOrder<UniformDistribution>&) const;

template std::vector<std::complex<double>>
QlmEval::eval<FisherDistribution>(const BondOrder<FisherDistribution>&) const;
