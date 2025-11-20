// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

#include "../data/Vec3.h"

namespace spatula { namespace optimize {
/**
 * @brief Base class for spatula optimizers.
 *
 * We use a state model where an optimizer is always either expecting an objective for a queried
 * point or to be queried for a point. To do the other * operation is an error and leads to an
 * exception.
 *
 * The optimizer exclusively optimizes over SO(3) or the space of 3D rotations. The class also
 * assumes that all generated points will lie on the unit 4D * hypersphere (or in other words the
 * quaternion is normalized).
 *
 * The optimizers uses a 3-vector, \f$ \nu \f$ to represent rotations in
 * \f$ SO(3) \f. The conversion to the axis-angle representation for
 * \f$ \nu \f$ is

 * \f$ \alpha = \frac{\nu}{||\nu||} \f$
 * \f$ \theta = ||\nu||. \f$

 *
 * Note: All optimizations are assumed to be minimizations. Multiply the objective function by -1 to
 * switch an maximization to a minimization.
 */
class Optimizer {
    public:
    /**
     * @brief Create an Optimizer. The only thing this does is set up the bounds.
     */
    Optimizer()
        : m_point(), m_objective(),
          m_best_point({0.0, 0.0, 0.0}, std::numeric_limits<double>::max()), m_count(0),
          m_need_objective(false)
    {
    }
    virtual ~Optimizer() = default;

    /// Get the next point to compute the objective for.
    data::Vec3 next_point()
    {
        if (m_need_objective) {
            throw std::runtime_error("Must record objective for new point first.");
        }
        internal_next_point();
        ++m_count;
        m_need_objective = true;
        return m_point;
    }
    /// Record the objective function's value for the last querried point.
    virtual void record_objective(double objective)
    {
        if (!m_need_objective) {
            throw std::runtime_error("Must get new point before recording objective.");
        }
        m_need_objective = false;
        m_objective = objective;
        if (objective < m_best_point.second) {
            m_best_point.first = m_point;
            m_best_point.second = objective;
        }
    }
    /// Returns whether or not convergence or termination conditions have been met.
    virtual bool terminate() const = 0;

    /// Get the current best point and the value of the objective function at that point.
    std::pair<data::Vec3, double> get_optimum() const
    {
        return m_best_point;
    }

    /// Create a clone of this optimizer
    virtual std::unique_ptr<Optimizer> clone() const = 0;

    /// Set the next point to compute the objective for to m_point.
    virtual void internal_next_point() = 0;

    unsigned int getCount() const
    {
        return m_count;
    }

    protected:
    /// The current point to evaluate the objective function for.
    data::Vec3 m_point;
    /// The last recorded objective function value.
    double m_objective;

    /// The best (as of yet) point computed.
    std::pair<data::Vec3, double> m_best_point;

    /// The number of iterations thus far.
    unsigned int m_count;

    /// A flag for which operation, next_point or record_objective, is allowed.
    bool m_need_objective;
};

/**
 * @brief Trampoline class for exposing Optimizer in Python.
 *
 * This shouldn't actually be used to extend the class but we need this to pass Optimizers through
 * from Python.
 */
class PyOptimizer : public Optimizer {
    public:
    using Optimizer::Optimizer;

    ~PyOptimizer() override = default;

    /// Get the next point to compute the objective for.
    void internal_next_point() override
    {
        PYBIND11_OVERRIDE_PURE(void, Optimizer, internal_next_point);
    }
    /// Record the objective function's value for the last querried point.
    void record_objective(double objective) override
    {
        PYBIND11_OVERRIDE(void, Optimizer, record_objective, objective);
    }
    /// Returns whether or not convergence or termination conditions have been met.
    bool terminate() const override
    {
        PYBIND11_OVERRIDE_PURE(bool, Optimizer, terminate);
    }

    /// Create a clone of this optimizer
    virtual std::unique_ptr<Optimizer> clone() const override
    {
        return std::make_unique<PyOptimizer>(*this);
    }
};
}} // namespace spatula::optimize
