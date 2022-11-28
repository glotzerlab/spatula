#pragma once
#include <complex>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../BondOrder.h"
#include "Util.h"

namespace py = pybind11;

namespace pgop { namespace util {
// TODO: pass normalization factor not m for generalizing.
/**
 * @brief Helper class to make computation of \f$ Q_{m}^{l} \f$ more efficient. The class upon
 * initialization computes weighted spherical harmonics according to the provided weights and
 * positions. Currently this expects a spherical surface Gauss-Legendre quadrature where m is the
 * order of the quadrature.
 */
class QlmEval {
    public:
    /**
     * @brief Create a QlmEval and pre-compute as much computation as possible.
     *
     * @param m the order of the Gauss-Legendre quadrature.
     * @param positions NumPy array of positions of shape \f$ (N_{quad}, 3) \f$.
     * @param positions NumPy array of quadrature weights of shape \f$ (N_{quad}) \f$.
     * @param positions NumPy array of spherical harmonics of shape \f$ (N_{lm}, N_{quad}) \f$. The
     * ordering of the first dimension is in accending order of \f$ l \f$ and \f$ m \f$.
     */
    QlmEval(unsigned int m,
            const py::array_t<double> positions,
            const py::array_t<double> weights,
            const py::array_t<std::complex<double>> ylms);

    /**
     * @brief For the provided bond order diagram compute the spherical harmonic expansion
     * coefficients. The method is templated on bond order type.
     *
     * We could use a base type and std::unique_ptr to avoid the template if desired in the future.
     *
     * @param bod the bond order diagram to use for evaluating the quadrature positions.
     * @returns the \f$ Q_{m}^{l} \f$ for the spherical harmonic expansion.
     */
    template<typename distribution_type>
    std::vector<std::complex<double>> eval(const BondOrder<distribution_type>& bod) const;

    /**
     * @brief For the provided bond order diagram compute the spherical harmonic expansion
     * coefficients in-place. The method is templated on bond order type.
     *
     * We could use a base type and std::unique_ptr to avoid the template if desired in the future.
     *
     * @param bod the bond order diagram to use for evaluating the quadrature positions.
     * @qlm_buf the buffer to place the \f$ Q_{m}^{l} \f$ for the spherical harmonic expansion in.
     */
    template<typename distribution_type>
    void eval(const BondOrder<distribution_type>& bod,
              std::vector<std::complex<double>>& qlm_buf) const;

    /// Get the number of unique combintations of \f$ l \f$ and \f$ m \f$.
    unsigned int getNlm() const;

    /// Get the maximum l value represented in the stored Ylms.
    unsigned int getMaxL() const;

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

    QlmBuf(size_t size);
};
}} // namespace pgop::util
