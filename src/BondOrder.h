#pragma once

#include <memory>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Util.h"

/**
 * @brief Null class to show the necessary interface for a spherical surface distribution. Given the
 * performance critical nature of the evaluation the bond order diagram, we should likely not move
 * this to an interface like approach as this would introduce a vtable and the need to access the
 * distribution on the heap.
 *
 * Notice that the position of a point in operator() is not given just the distance from the
 * distribution mean.
 *
 * This remains as a pure reminder of the interface expected of distributions.
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
     * @brief Compute the distribution's value at a distance of theta radians away from the mean.
     * All distributions that are radially symmetric (on the sphere's surface) should be reducable
     * to this approach.
     */
    double operator()(double theta) const;
};

/**
 * @brief Represents a uniform and normalized distribution centered at a given position on the unit
 * sphere. The distribuition is as follows,
 * \f{align*}
 *     p(\theta) &= \frac{1}{2 * \pi * (1 - \cos{\theta_{max}})} \quad \if\ \theta \le \theta_{max}
 * \\
 *               &= 0 \quad \text{else}
 * \f{align*}
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

    double operator()(double theta) const;

    private:
    /// max_theta The distance in radian from the mean that is non-zero.
    double m_max_theta;
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

    double operator()(double theta) const;

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
 * @tparam distribution_type A type that matches the interface of SphereSurfaceDistribution.
 */
template<typename distribution_type> class BondOrder {
    public:
    /**
     * @brief Create a BondOrder<distribution_type> object from a distribution and normalized
     * position vectors.
     *
     * @param The distribution to be centered at the given neighbor vectors.
     * @param The normalized (lie on the unit sphere) neighbor vectors. These serve as the mean for
     * the \f$ N \f$ distributions on the bond order diagram.
     */
    BondOrder(distribution_type dist, const std::vector<Vec3>& positions);

    // Assumes points are on the unit sphere
    std::vector<double> operator()(const std::vector<Vec3>& points) const;

    private:
    /**
     * @brief Compute the bond order diagram at a given point.
     *
     * @param point A point on the unit sphere in Cartesian coordinates.
     */
    double single_call(const Vec3& point) const;

    /// The distribution to use for all provided neighbor vectors.
    distribution_type m_dist;
    /// The normalized neighbor vectors for the bond order diagram.
    std::vector<Vec3> m_positions;
    /// The normalization constant @c 1 / static_cast<double>(m_positions.size()).
    double m_normalization;
};
