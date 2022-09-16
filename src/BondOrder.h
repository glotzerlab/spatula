#pragma once

#include <vector>

#include "data/Vec3.h"

namespace pgop {
/**
 * @brief Null class to show the necessary interface for a spherical surface distribution. Given the
 * performance critical nature of the evaluation the bond order diagram, we should likely not move
 * this to an interface like approach as this would introduce a vtable and the need to access the
 * distribution on the heap. However, when we can use C++20 we can use a compile-time interface
 * using concepts we can flesh this out into the required concept for BondOrder.
 *
 * Two options for computing the distribution exist:
 *   1. Compute the distribution from the angle between the point and the mean.
 *   2. The dot product of the mean with the point which when the arccos is taken produces the angle
 *      from 1. Therefore, this can be faster if the distribtion can be expressed without the angle.
 */
class SphereSurfaceDistribution {
    public:
    /**
     * Each distribution must have a param_type typedef which has a corresponding single argument
     * constructor.
     */
    using param_type = double;
    SphereSurfaceDistribution(param_type param);

    /**
     * @brief Compute the distribution's value.
     *
     * @param x The meaning of x is given by the use_theta static member. If true, x is the angle
     * between the mean and the point. If false, x is the dot product of the mean and the point.
     */
    double operator()(double x) const;

    /// Whether to use theta in operator().
    static const bool use_theta = false;
};

/**
 * @brief Represents a uniform and normalized distribution centered at a given position on the unit
 * sphere. The distribuition is as follows,
 * \f{align*}
 *     p(\theta) &= \frac{1}{2 * \pi * (1 - \cos{\theta_{max}})} \quad \if\ \theta \le \theta_{max}
 * \\
 *               &= 0 \quad \text{else}
 * \f{align*}
 *
 * To speed up the performance we use the fact that \f$ \cos{x} \f$ is monotonically decreasing in
 * the domain \f$ [0, \pi] \f$ to transform the check to \f$ \mu \cdot x < cos{\theta_{max}} \f$
 * precomputing the cosign.
 */
class UniformDistribution {
    public:
    using param_type = double;

    /**
     * @brief Create a UniformDistribution.
     *
     * @param max_theta The distance in radian from the mean that is non-zero.
     */
    UniformDistribution(double max_theta);

    double operator()(double x) const;

    static const bool use_theta = false;

    private:
    /**
     * The value which if the dot product (x in operator()) of the point and the mean is equal to
     * or lower than the distribution is 0.
     */
    double m_threshold;
    /// The normalization constant for the distribution.
    double m_prefactor;
};

/**
 * @brief Represents a von-Mises-Fisher distribution centered at a given position on the unit
 * sphere. The distribuition is as follows,
 * \f[
 *     p(\theta) = \frac{e^{\kappa \cos{\theta}}}{2 * \pi * (e^{\kappa} - e^{-\kappa})}
 * \f]
 */
class FisherDistribution {
    public:
    using param_type = double;

    /**
     * Create a Fisher distribution.
     *
     * @param kappa the concentration parameter of the distribution. Larger values result in a more
     * concentrated (tighter) distribution.
     */
    FisherDistribution(double kappa);

    double operator()(double x) const;

    static const bool use_theta = false;

    private:
    /**
     * kappa the concentration parameter of the distribution. Larger values result in a more
     * concentrated (tighter) distribution.
     */
    double m_kappa;
    /// The normalization constant for the distribution.
    double m_prefactor;
};

/**
 * @brief Representation of a bond order diagram where each point in the diagram is represented by a
 * provided distribution. The class's main function is to compute the bond order diagram's value at
 * various points on the unit sphere.
 *
 * All computation and variables are in Cartesian coordinates as this simplifies the math and
 * increases performance.
 *
 * WARNING: This class stores references to std::vectors. This means that care must be taken that
 * the vectors outlive the object.
 *
 * @tparam distribution_type A type that matches the interface of SphereSurfaceDistribution.
 */
template<typename distribution_type> class BondOrder {
    public:
    /**
     * @brief Create a BondOrder<distribution_type> object from a distribution and normalized
     * position vectors.
     *
     * @param dist The distribution to be centered at the given neighbor vectors.
     * @param positions The normalized (lie on the unit sphere) neighbor vectors. These serve as the
     * mean for the \f$ N \f$ distributions on the bond order diagram.
     * @param weights The weights to use for each position. Should be the same size as positions.
     */
    BondOrder(distribution_type dist, const std::vector<data::Vec3>& positions, const std::vector<double>& weights);

    // Assumes points are on the unit sphere
    std::vector<double> operator()(const std::vector<data::Vec3>& points) const;

    private:
    /**
     * @brief Compute the bond order diagram at a given point.
     *
     * @param point A point on the unit sphere in Cartesian coordinates.
     */
    inline double single_call(const data::Vec3& point) const;

    /// The distribution to use for all provided neighbor vectors.
    distribution_type m_dist;
    /// The normalized neighbor vectors for the bond order diagram.
    const std::vector<data::Vec3>& m_positions;
    /// The weights for the points on the bond order diagram.
    const std::vector<double>& m_weights;
    /// The normalization constant @c 1 / std::reduce(m_weights).
    double m_normalization;
};
} // End namespace pgop
