#pragma once

#include <memory>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>

#include "../data/Vec3.h"

namespace nb = nanobind;

namespace pgop { namespace optimize {
/**
 * @brief Base class for pgop optimizers.
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
    Optimizer();
    virtual ~Optimizer() = default;

    /// Get the next point to compute the objective for.
    data::Vec3 next_point();
    /// Record the objective function's value for the last querried point.
    virtual void record_objective(double);
    /// Returns whether or not convergence or termination conditions have been met.
    virtual bool terminate() const = 0;

    /// Get the current best point and the value of the objective function at that point.
    std::pair<data::Vec3, double> get_optimum() const;

    /// Create a clone of this optimizer
    virtual std::unique_ptr<Optimizer> clone() const = 0;

    /// Set the next point to compute the objective for to m_point.
    virtual void internal_next_point() = 0;

    unsigned int getCount() const;

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
    NB_TRAMPOLINE(Optimizer, 4);
    public:
    using Optimizer::Optimizer;

    ~PyOptimizer() override = default;

    /// Get the next point to compute the objective for.
    void internal_next_point() override;
    /// Record the objective function's value for the last querried point.
    void record_objective(double) override;
    /// Returns whether or not convergence or termination conditions have been met.
    bool terminate() const override;

    /// Create a clone of this optimizer
    virtual std::unique_ptr<Optimizer> clone() const override;
};

void export_base_optimize(nb::module_& m);

}} // namespace pgop::optimize
