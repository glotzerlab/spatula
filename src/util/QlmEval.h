// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "../BondOrder.h"
#include "Util.h"

namespace spatula { namespace util {
// TODO: pass normalization factor not m for generalizing.
/**
 * @brief Helper class to make computation of \f$ Q_{m}^{l} \f$ more efficient.
 *
 * The class upon initialization computes weighted spherical harmonics according to the provided
 * weights and positions. Currently this expects a spherical surface Gauss-Legendre quadrature where
 * m is the order of the quadrature.
 */
class QlmEval {
    public:
    /**
     * @brief Create a QlmEval and pre-compute as much computation as possible.
     *
     * @param m the order of the Gauss-Legendre quadrature.
     * @param n_points The number of quadrature points.
     * @param n_lms The number of spherical harmonic moments.
     * @param positions_data pointer to an array of positions of shape \f$ (N_{quad}, 3) \f$.
     * @param weights_data pointer to an array of quadrature weights of shape \f$ (N_{quad}) \f$.
     * @param ylms_data pointer to an array of spherical harmonics of shape \f$ (N_{lm}, N_{quad}) \f$.
     */
    QlmEval(unsigned int m,
            size_t n_points,
            size_t n_lms,
            const double* positions_data,
            const double* weights_data,
            const std::complex<double>* ylms_data)
        : m_n_lms(n_lms), m_max_l(0), m_n_points(n_points), m_positions(),
          m_weighted_ylms()
    {
        unsigned int count = 1;
        while (count != m_n_lms) {
            ++m_max_l;
            count += 2 * m_max_l + 1;
        }
        m_weighted_ylms.reserve(m_n_lms);
        const double normalization = 1.0 / (4.0 * static_cast<double>(m));
        for (size_t lm {0}; lm < m_n_lms; ++lm) {
            auto ylm = std::vector<std::complex<double>>();
            ylm.reserve(m_n_points);
            for (size_t i {0}; i < m_n_points; ++i) {
                ylm.emplace_back(normalization * weights_data[i]
                                 * ylms_data[lm * m_n_points + i]);
            }
            m_weighted_ylms.emplace_back(ylm);
        }
        m_positions.reserve(m_n_points);
        for (size_t i {0}; i < m_n_points; ++i) {
            m_positions.emplace_back(&positions_data[i * 3]);
        }
    }

    /**
     * @brief For the provided bond order diagram compute the spherical harmonic expansion
     * coefficients. The method is templated on bond order type.
     *
     * We could use a base type and std::unique_ptr to avoid the template if desired in the future.
     * This would be slower though.
     *
     * @param bod the bond order diagram to use for evaluating the quadrature positions.
     * @returns the \f$ Q_{m}^{l} \f$ for the spherical harmonic expansion.
     */
    template<typename distribution_type>
    std::vector<std::complex<double>> eval(const BondOrder<distribution_type>& bod) const
    {
        std::vector<std::complex<double>> qlms;
        qlms.reserve(m_n_lms);
        eval(bod, qlms);
        return qlms;
    }

    /**
     * @brief For the provided bond order diagram compute the spherical harmonic expansion
     * coefficients in-place. The method is templated on bond order type.
     *
     * We could use a base type and std::unique_ptr to avoid the template if desired in the future.
     * Though this would make the conputation slower.
     *
     * @param bod the bond order diagram to use for evaluating the quadrature positions.
     * @qlm_buf the buffer to place the \f$ Q_{m}^{l} \f$ for the spherical harmonic expansion in.
     */
    template<typename distribution_type>
    void eval(const BondOrder<distribution_type>& bod,
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

    /// Get the number of unique combintations of \f$ l \f$ and \f$ m \f$.
    unsigned int getNlm() const
    {
        return m_n_lms;
    }

    /// Get the maximum l value represented in the stored Ylms.
    unsigned int getMaxL() const
    {
        return m_max_l;
    }

    private:
    /// Number of unique combintations of \f$ l \f$ and \f$ m \f$.
    unsigned int m_n_lms;

    /// Maximum l computed from the input size of Ylms
    unsigned int m_max_l;
    // TODO just make this a fuction this->m_positions.size()
    /// Number of points in quadrature.
    unsigned int m_n_points;
    /// The quadrature points.
    std::vector<data::Vec3> m_positions;
    /// Precomputed weighted ylms of the provided quadrature and normalization.
    std::vector<std::vector<std::complex<double>>> m_weighted_ylms;
};

/**
 * @brief A class to make passing buffers for spherical harmonic expansions and their symmetrized
 * expensions easier.
 */
struct QlmBuf {
    /// base values
    std::vector<std::complex<double>> qlms;
    /// symmetrized values
    std::vector<std::complex<double>> sym_qlms;

    QlmBuf(size_t size) : qlms(), sym_qlms()
    {
        qlms.reserve(size);
        sym_qlms.reserve(size);
    }
};
}} // namespace spatula::util
