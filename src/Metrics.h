#pragma once

#include <complex>
#include <vector>

/**
 * @brief Compute a weighted p-norm of a N dimensional vector.
 *
 * The weighted computation is
 * \f[
 *   \left(\sum_{i = 1}^{N}{w_i x_i^p} \right)^{1 / p} / \sum_{i=1}^{N}{w_i}.
 * \f]
 *
 * Assumes (does not check) that weights and provided vectors are of the same length. More
 * accurately m_weights.size() >= vector.size() is required.
 *
 * This class is used to allow runtime polymorphism with respect to the PGOP scoring of neighborhood
 * symmetrization.
 */
class WeightedPNormBase {
    public:
    /// Create an unweighted WeightedPNormBase object.
    WeightedPNormBase();

    /**
     * @brief Create a WeightedPNormBase object which is weighted in the dimensions.
     *
     * @param weights the weights to apply to each dimension of the vectors given to operator(). The
     * vector needs to be the same size as the vectors to be passed to operator(). This is not
     * checked.
     */
    WeightedPNormBase(const std::vector<double>& weights);

    /// Compute the weighted p-norm for provided vector.
    virtual double operator()(const std::vector<double>& vector) const = 0;

    protected:
    /// Weights for the weighted p-norm computation
    const std::vector<double> m_weights;
    /// Constant (summation of weight) to normalize weighted p-norm computation.
    const double m_normalization;
};

/**
 * @brief Provide the logic for a given weighted p-norm computation.
 *
 * All documentation from WeightedPNormBase applies here.
 *
 * @tparam p the p-norm to use.
 */
template<unsigned int p> class WeightedPNorm : public WeightedPNormBase {
    public:
    WeightedPNorm();
    WeightedPNorm(const std::vector<double>& weights);

    double operator()(const std::vector<double>& vector) const override;
};

/**
 * @brief compute the Pearson correlation between \f$ B \f$ and \f$ B_{sym, i}, \forall i \f$ where
 * \f$ B \f$ is the bond order diagram and \f$ i \f$ is the point group. We do this through the
 * spherical harmonic expansion of \f$ B \f$ and its symmetrized expansions through the coefficients
 * \f$ Q_{m}^{l} \f$.
 *
 * The implementation uses some tricks to make the computation as efficient as possible compared to
 * a standard corrlation computation.
 *
 * @param qlms The coefficents for the spherical harmonic expansion of \f$ B \f$.
 * @param sym_qlms The coefficents for all symmetrized spherical harmonic expansion of
 * \f$ B_{sym, i} \f$ where \f$ i \f$ indicates which point group symmetrization has been done.
 * @returns A vector of the Pearson correlation for each point group symmetrization.
 */
std::vector<double> covariance(const std::vector<std::complex<double>>& qlms,
                               const std::vector<std::vector<std::complex<double>>>& sym_qlms);
