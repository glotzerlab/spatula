#pragma once

#include <memory>
#include <utility>
#include <vector>

namespace pgop { namespace optimize {
/**
 * @brief Base class for pgop optimizers. We use a state model where an optimizer is always either
 * expecting an objective for a queried point or to be queried for a point. To do the other
 * operation is an error and leads to an exception.
 *
 * The optimizer also includes support for bounded optimization domains through minimum and maximum
 * bounds. Given that many solvers do not natively support bounds on the solution's domain, users
 * should be careful in using this feature, and when doing so attempt to start solvers away from the
 * bounds.
 *
 * Note: All optimizations are assumed to be minimizations. Multiply the objective function by -1 to
 * switch an maximization to a minimization.
 */
class Optimizer {
    public:
    /**
     * @brief Create an Optimizer. The only thing this does is set up the bounds.
     *
     * @param min_bounds The minimum value allowed for each dimension. Set to
     * -std::numeric_limits<double>::infinity() for no minimum bounds.
     * @param max_bounds The maximum value allowed for each dimension. Set to
     * std::numeric_limits<double>::infinity() for no maximum bounds.
     */
    Optimizer(const std::vector<double>& min_bounds, const std::vector<double>& max_bounds);
    virtual ~Optimizer() = default;

    /// Get the next point to compute the objective for.
    std::vector<double> next_point();
    /// Record the objective function's value for the last querried point.
    virtual void record_objective(double);
    /// Returns whether or not convergence or termination conditions have been met.
    virtual bool terminate() const = 0;

    /// Get the current best point and the value of the objective function at that point.
    std::pair<std::vector<double>, double> get_optimum() const;

    /// Create a clone of this optimizer
    virtual std::unique_ptr<Optimizer> clone() const = 0;

    /// Potentially modify optimizer for given particle (e.g. random algorithms).
    virtual void specialize(unsigned int particle_index);

    /// Set the next point to compute the objective for to m_point.
    virtual void internal_next_point() = 0;

    unsigned int getCount() const;

    const std::vector<double>& getMinBounds() const;
    const std::vector<double>& getMaxBounds() const;

    protected:
    /// Take a point and wrap it to the nearest within bounds point.
    void clip_point(std::vector<double>& point);

    /// The minimum value allowed for each dimension.
    const std::vector<double> m_min_bounds;
    /// The maximum value allowed for each dimension.
    const std::vector<double> m_max_bounds;

    /// The current point to evaluate the objective function for.
    std::vector<double> m_point;
    /// The last recorded objective function value.
    double m_objective;

    /// The best (as of yet) point computed.
    std::pair<std::vector<double>, double> m_best_point;

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
    void internal_next_point() override;
    /// Record the objective function's value for the last querried point.
    void record_objective(double) override;
    /// Returns whether or not convergence or termination conditions have been met.
    bool terminate() const override;

    /// Create a clone of this optimizer
    virtual std::unique_ptr<Optimizer> clone() const override;
};

}} // namespace pgop::optimize
