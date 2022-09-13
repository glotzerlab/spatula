#pragma once

#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Base class for pgop optimizers. We use a state model where an optimizer is always either
 * expecting an objective for a queried point or to be queried for a point. To do the other
 * operation is an error and leads to an exception.
 */
class Optimizer {
    public:
    Optimizer(const std::vector<double>& min_bounds, const std::vector<double>& max_bounds);

    virtual std::vector<double> next_point() = 0;
    virtual void record_objective(double);
    virtual bool terminate() const = 0;
    virtual std::pair<std::vector<double>, double> get_optimum() const = 0;

    /// Create a clone of this optimizer
    virtual std::unique_ptr<Optimizer> clone() const = 0;

    protected:
    void clip_point(std::vector<double>& point);

    const std::vector<double> m_min_bounds;
    const std::vector<double> m_max_bounds;

    std::vector<double> m_point;
    double m_objective;

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
    BruteForce(const std::vector<std::vector<double>>& points,
               const std::vector<double>& min_bounds,
               const std::vector<double>& max_bounds);

    void record_objective(double) override;
    std::vector<double> next_point() override;
    bool terminate() const override;
    std::pair<std::vector<double>, double> get_optimum() const override;
    std::unique_ptr<Optimizer> clone() const override;


    private:
    std::vector<std::vector<double>> m_points;
    size_t m_cnt;

    std::vector<double> m_best_point;
    double m_best_objective;
};

struct NelderMeadParams {
    double alpha;
    double gamma;
    double rho;
    double sigma;

    NelderMeadParams(double alpha_, double gamma_, double rho_, double sigma_);
};

class RollingStd {
    public:
    RollingStd();
    RollingStd(const std::vector<double>& values);
    void update(double new_value, double old_value);
    double std() const;
    double mean() const;

    private:
    double m_mean;
    double m_var;
    double m_n;
};

double compute_distance(const std::vector<double>& a, const std::vector<double>& b);

class OrderedSimplex {
    public:
    OrderedSimplex(unsigned int dim);

    void add(const std::vector<double>& point, const double objective);

    const std::vector<double>& get_point(size_t index) const;

    size_t size() const;

    double get_objective(size_t index) const;

    double get_objective_std() const;
    double get_objective_mean() const;
    double get_min_dist() const;

    std::vector<double> compute_centroid() const;

    const std::pair<std::vector<double>, double>& operator[](size_t index) const;

    private:
    void complete_initialization();
    void update_min_distance(const std::vector<double>& new_point);

    unsigned int m_dim;
    std::vector<std::pair<std::vector<double>, double>> m_points;
    RollingStd m_rolling_std;
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
