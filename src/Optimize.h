#pragma once

#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;

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

    /// Get the next point to compute the objective for.
    virtual std::vector<double> next_point() = 0;
    /// Record the objective function's value for the last querried point.
    virtual void record_objective(double);
    /// Returns whether or not convergence or termination conditions have been met.
    virtual bool terminate() const = 0;
    /// Get the current best point and the value of the objective function at that point.
    virtual std::pair<std::vector<double>, double> get_optimum() const = 0;

    /// Create a clone of this optimizer
    virtual std::unique_ptr<Optimizer> clone() const = 0;

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

    /// Get the next point to compute the objective for.
    std::vector<double> next_point() override;
    /// Record the objective function's value for the last querried point.
    void record_objective(double) override;
    /// Returns whether or not convergence or termination conditions have been met.
    bool terminate() const override;
    /// Get the current best point and the value of the objective function at that point.
    std::pair<std::vector<double>, double> get_optimum() const override;

    /// Create a clone of this optimizer
    virtual std::unique_ptr<Optimizer> clone() const override;
};

/**
 * @brief An Optimizer that just tests prescribed points. The optimizer picks the best point out of
 * all provided test points.
 */
class BruteForce : public Optimizer {
    public:
    /**
     * @brief Create an BruteForce.
     *
     * All parameters are expected to have matching dimensions.
     *
     * @param points The points to test. Expected sizes are \f$ (N_{brute}, N_{dim} \f$.
     * @param min_bounds The minimum value allowed for each dimension. Set to
     * -std::numeric_limits<double>::infinity() for no minimum bounds. Expected dimension is
     * \f$ N_{dim} \f$.
     * @param max_bounds The maximum value allowed for each dimension. Set to
     * std::numeric_limits<double>::infinity() for no maximum bounds. Expected dimension is
     * \f$ N_{dim} \f$.
     */
    BruteForce(const std::vector<std::vector<double>>& points,
               const std::vector<double>& min_bounds,
               const std::vector<double>& max_bounds);

    void record_objective(double) override;
    std::vector<double> next_point() override;
    bool terminate() const override;
    std::pair<std::vector<double>, double> get_optimum() const override;
    std::unique_ptr<Optimizer> clone() const override;

    private:
    /// The set of points to evaluate.
    std::vector<std::vector<double>> m_points;
    /// The current iteration of the optimizer.
    size_t m_cnt;

    /// Current optimum point.
    std::vector<double> m_best_point;
    /// Current optimum objective.
    double m_best_objective;
};

/// Struct of Nelder-Mead optimization hyper-parameters.
struct NelderMeadParams {
    /// Multiplicative factor in reflects (determines the extend of reflection steps).
    double alpha;
    /// Multiplicative factor in expanding (determines the extend of expansion steps).
    double gamma;
    /// Multiplicative factor in contracting (determines the extend of contraction steps).
    double rho;
    /// Multiplicative factor in shrinking (determines the extend of shrink steps).
    double sigma;

    /// Construct a NelderMeadParams from explicitly provided values.
    NelderMeadParams(double alpha_, double gamma_, double rho_, double sigma_);
    // TODO add default constructor that uses the suggested values from wikipedia
};

/**
 * @brief Helper class that efficiently computes the rolling first (mean) and second (variance)
 * moments of a set of changing points. The objects window size is determined at initialization and
 * is immutable from that point on.
 *
 * This class significant reduces the time that convergence/termination standard deviation or mean
 * based conditions take to evaluate.
 */
class RollingStd {
    public:
    /// @brief Construct a null RollingStd. This is effectively useless.
    RollingStd();
    /**
     * @brief Construct a RollingStd from an initial set of points (window).
     *
     * @params values The \f$ N \f$ initial points to initialize the object with. This fixes the
     * window size of the class as well.
     */
    RollingStd(const std::vector<double>& values);
    /// Update moments with new_value while removing contribution of old_value.
    void update(double new_value, double old_value);
    /// Current window's standard deviation.
    double std() const;
    /// Current window's mean.
    double mean() const;

    private:
    /// Current mean.
    double m_mean;
    /// Current variance.
    double m_var;
    /// Window size.
    double m_n;
};

/// Convenience function that computes the \f$ l_2 \f$ distance between two same size vectors.
double compute_distance(const std::vector<double>& a, const std::vector<double>& b);

/**
 * @brief A container for an ordered simplex of a given dimension. The class contains a vector of
 * \f$ N_{dim} + 1 \f$ points sorted by a given value (objective function in NelderMead).
 *
 * The class uses add to potenially add a new point.
 *
 * The class also computes the mean and standard deviation of the simplex points using RollingStd.
 * This removes some of the classes purity, but practically increases performance of the
 * optimization.
 */
class OrderedSimplex {
    public:
    /// Construct an OrderedSimplex of dimension dim.
    OrderedSimplex(unsigned int dim);

    /**
     * @brief Attempt to add point to simplex.
     *
     * @param point The point to add.
     * @param objective The value to rank against other simplex points. If the objective is too high
     * the point will not be added to the simplex.
     */
    void add(const std::vector<double>& point, const double objective);

    /// Get the point at index from the sorted simplex points.
    const std::vector<double>& get_point(size_t index) const;

    /// The size of the simplex \f$ N_{dim] + 1 \f$.
    size_t size() const;

    /// Get the objective/value of the point at index in the sorted simplex.
    double get_objective(size_t index) const;

    /// Get the standard deviation of the provided objectives/values for the points.
    double get_objective_std() const;
    /// Get the mean of the provided objectives/values for the points.
    double get_objective_mean() const;
    /// Get the minimum Euclidean distance between points.
    double get_min_dist() const;

    /// Compute the centroid of the first \f$ N_{dim} \f$ points.
    std::vector<double> compute_centroid() const;

    /// Get the point and value/objective at index in the sorted simplex.
    const std::pair<std::vector<double>, double>& operator[](size_t index) const;

    private:
    /// Helper function to do the initial statistic and sorting for the simplex.
    void complete_initialization();
    /// Update the m_min_dist with a new point in the simplex.
    void update_min_distance(const std::vector<double>& new_point);

    /// The dimension of the simplex
    unsigned int m_dim;
    /// The current points of the simplex
    std::vector<std::pair<std::vector<double>, double>> m_points;
    /// Rolling statistic computer
    RollingStd m_rolling_std;
    /// The current minimum distance between any two points in the simplex.
    double m_min_dist;
};

class NelderMead : public Optimizer {
    public:
    NelderMead(NelderMeadParams params,
               const std::vector<std::vector<double>>& initial_simplex,
               const std::vector<double>& min_bounds,
               const std::vector<double>& max_bounds,
               unsigned int max_iter,
               double m_dist_tol,
               double m_std_tol);

    std::vector<double> next_point() override;
    bool terminate() const override;
    std::pair<std::vector<double>, double> get_optimum() const override;
    std::unique_ptr<Optimizer> clone() const override;

    private:
    enum Stage {
        NEW_SIMPLEX = 0,
        REFLECT = 1,
        EXPAND = 2,
        OUTSIDE_CONTRACT = 3,
        INSIDE_CONTRACT = 4,
    };

    std::vector<double> reflect();

    std::vector<double> expand();

    std::vector<double> outside_contract();

    std::vector<double> inside_contract();

    std::vector<double> shrink();

    Stage m_stage;
    const NelderMeadParams m_params;
    const unsigned int m_dim;
    OrderedSimplex m_current_simplex;

    unsigned int m_max_iter;
    unsigned int m_iter;
    double m_dist_tol;
    double m_std_tol;

    std::pair<std::vector<double>, double> m_last_reflect;
    size_t m_new_simplex_index;
    std::vector<std::vector<double>> m_new_simplex;
};

void export_optimize(py::module& m);
