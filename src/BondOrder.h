// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include "data/Vec3.h"
#include "util/Util.h"

#ifdef _MSC_VER
#define M_PI 3.14159265358979323846
#endif

namespace spatula {
/**
 * @brief Concept to show the necessary interface for a spherical surface distribution. Given the
 * performance critical nature of the evaluation the bond order diagram, we should use concepts and
 * not inheritance.
 *
 * Two options for computing the distribution (operator(double x)) exist:
 *   1. Compute the distribution from the angle between the point and the mean.
 *   2. The dot product of the mean with the point which when the arccos is taken produces the angle
 *      from 1. Therefore, this can be faster if the distribtion can be expressed without the angle.
 */
/* template<typename T> */
/* concept SphereSurfaceDistribution = requires(T d, double x) */
/* { */
/*     // Require a type alias for the constructor's single argument's type. */
/*     typename T::param_type; */
/*     // Require Distribution::operator()(double x) -> double or float. */
/*     {std::as_const(d)(x)} -> std::floating_point; */
/*     // Require constructor of only param_type. */
/* } && std::constructible_from<typename T::param_type> && */
/*     // Require a static member use_theta which determines the value passed to operator(). */
/*     std::same_as<decltype(T::use_theta), const bool>; */

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
    inline UniformDistribution(param_type max_theta)
        : m_threshold(std::cos(max_theta)), m_prefactor(1 / (2 * M_PI * (1 - std::cos(max_theta))))
    {
    }

    /**
     * @brief Return the value of the distribution at the given point.
     *
     * @param x The distance from the mean to evaluate the distribution at.
     */
    inline double operator()(double x) const
    {
        return x > m_threshold ? m_prefactor : 0;
    }

    /// operator() uses the distance to evaluate
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
    inline FisherDistribution(param_type kappa)
        : m_kappa(kappa), m_prefactor(kappa / (2 * M_PI * (std::exp(kappa) - std::exp(-kappa))))
    {
    }

    /**
     * @brief Return the value of the distribution at the given point.
     *
     * @param x The distance from the mean to evaluate the distribution at.
     */
    inline double operator()(double x) const
    {
        return m_prefactor * std::exp(m_kappa * x);
    }

    /// operator() uses the distance to evaluate
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

// When updating to C++20, use SphereSurfaceDistribution instead of typename below.

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
 * @tparam distribution_type A type that matches the SphereSurfaceDistribution concept.
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
    inline BondOrder(distribution_type dist,
                     std::span<const data::Vec3> positions,
                     std::span<const double> weights)
        : m_dist(dist), m_positions(positions), m_weights(weights),
          m_normalization(1.0 / std::reduce(m_weights.begin(), m_weights.end()))
    {
    }

    // Assumes points are on the unit sphere
    inline std::vector<double> operator()(std::span<const data::Vec3> points) const
    {
        auto bo = std::vector<double>();
        bo.reserve(points.size());
        std::transform(points.begin(),
                       points.end(),
                       std::back_inserter(bo),
                       [this](const auto& point) { return this->single_call(point); });
        return bo;
    }

    private:
    /**
     * @brief Compute the bond order diagram at a given point.
     *
     * @param point A point on the unit sphere in Cartesian coordinates.
     */
    inline double single_call(const data::Vec3& point) const
    {
        double sum_correction = 0;
        // Get the unweighted contribution from each distribution lazily.
        auto single_contributions = std::vector<double>();
        single_contributions.resize(m_positions.size());
        std::transform(m_positions.begin(),
                       m_positions.end(),
                       single_contributions.begin(),
                       [this, &point](const auto& p) -> double {
                           if constexpr (distribution_type::use_theta) {
                               return this->m_dist(util::fast_angle_eucledian(p, point));
                           } else {
                               return this->m_dist(p.dot(point));
                           }
                       });
        // Normalize the value and weight the contributions.
        return m_normalization
               * std::transform_reduce(
                   single_contributions.begin(),
                   single_contributions.end(),
                   m_weights.begin(),
                   0.0,
                   // Use Kahan summation to improve accuracy of the summation of small
                   // numbers.,
                   [&sum_correction](const auto& sum, const auto& y) -> double {
                       auto addition = y - sum_correction;
                       const auto new_sum = sum + addition;
                       sum_correction = new_sum - sum - addition;
                       return new_sum;
                   },
                   std::multiplies<>());
    }

    /// The distribution to use for all provided neighbor vectors.
    distribution_type m_dist;
    /// The normalized neighbor vectors for the bond order diagram.
    std::span<const data::Vec3> m_positions;
    /// The weights for the points on the bond order diagram.
    std::span<const double> m_weights;
    /// The normalization constant @c 1 / std::reduce(m_weights).
    double m_normalization;
};
} // End namespace spatula
