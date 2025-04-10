// Copyright (c) 2010-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <algorithm>
#include <cmath>
#include <iterator>

#include "QlmEval.h"

namespace spatula { namespace util {
QlmEval::QlmEval(unsigned int m,
                 const py::array_t<double> positions,
                 const py::array_t<double> weights,
                 const py::array_t<std::complex<double>> ylms)
    : m_n_lms(ylms.shape(0)), m_max_l(0), m_n_points(ylms.shape(1)), m_positions(),
      m_weighted_ylms()
{
    unsigned int count = 1;
    while (count != m_n_lms) {
        ++m_max_l;
        count += 2 * m_max_l + 1;
    }
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
    for (size_t i {0}; i < static_cast<size_t>(positions.shape(0)); ++i) {
        m_positions.emplace_back(u_positions.data(i, 0));
    }
}

unsigned int QlmEval::getMaxL() const
{
    return m_max_l;
}

template<typename distribution_type>
void QlmEval::eval(const BondOrder<distribution_type>& bod,
                   std::vector<std::complex<double>>& qlm_buf) const
{
    qlm_buf.clear();
    qlm_buf.reserve(m_n_lms);
    const auto B_quad = bod(m_positions);
    std::transform(m_weighted_ylms.begin(),
                   m_weighted_ylms.end(),
                   std::back_insert_iterator(qlm_buf),
                   [&B_quad](const auto& w_ylm) {
                       std::complex<double> dot = 0;
                       size_t i = 0;
                       // Attempt to unroll loop for improved performance.
                       for (; i + 10 < w_ylm.size(); i += 10) {
                           // Simple summation seems to work here unlike in the BondOrder<> classes.
                           dot += B_quad[i] * w_ylm[i] + B_quad[i + 1] * w_ylm[i + 1]
                                  + B_quad[i + 2] * w_ylm[i + 2] + B_quad[i + 3] * w_ylm[i + 3]
                                  + B_quad[i + 4] * w_ylm[i + 4] + B_quad[i + 5] * w_ylm[i + 5]
                                  + B_quad[i + 6] * w_ylm[i + 6] + B_quad[i + 7] * w_ylm[i + 7]
                                  + B_quad[i + 8] * w_ylm[i + 8] + B_quad[i + 9] * w_ylm[i + 9];
                       }
                       for (; i < w_ylm.size(); ++i) {
                           dot += B_quad[i] * w_ylm[i];
                       }
                       return dot;
                   });
}

template<typename distribution_type>
std::vector<std::complex<double>> QlmEval::eval(const BondOrder<distribution_type>& bod) const
{
    std::vector<std::complex<double>> qlms;
    qlms.reserve(m_n_lms);
    eval(bod, qlms);
    return qlms;
}

unsigned int QlmEval::getNlm() const
{
    return m_n_lms;
};

template std::vector<std::complex<double>>
QlmEval::eval<UniformDistribution>(const BondOrder<UniformDistribution>&) const;
template void QlmEval::eval<UniformDistribution>(const BondOrder<UniformDistribution>&,
                                                 std::vector<std::complex<double>>&) const;

template std::vector<std::complex<double>>
QlmEval::eval<FisherDistribution>(const BondOrder<FisherDistribution>&) const;
template void QlmEval::eval<FisherDistribution>(const BondOrder<FisherDistribution>&,
                                                std::vector<std::complex<double>>&) const;

QlmBuf::QlmBuf(size_t size) : qlms(), sym_qlms()
{
    qlms.reserve(size);
    sym_qlms.reserve(size);
}

}} // namespace spatula::util
