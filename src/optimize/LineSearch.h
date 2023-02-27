#pragma once

#include <random>

#include <pybind11/pybind11.h>

#include "Optimize.h"

namespace pgop { namespace optimize {

class LineSearch : public Optimizer {
    public:
    LineSearch(const data::Vec3& initial_point,
               unsigned int max_iter,
               double initial_jump,
               double learning_rate,
               double tol);
    ~LineSearch() override = default;
    /// Returns whether or not convergence or termination conditions have been met.
    bool terminate() const override;
    /// Create a clone of this optimizer
    std::unique_ptr<Optimizer> clone() const override;

    /// Set the next point to compute the objective for to m_point.
    void internal_next_point() override;

    private:
    enum Stage { INITIALIZE = 1, GRADIENT = 2, SEARCH = 4 };
    void step();
    void initialize();
    void findGradient();
    void searchAlongGradient();

    // Hyperparameters
    unsigned int m_max_iter;
    double m_learning_rate;
    double m_initial_jump;
    double m_tol;

    // State variables
    // General
    Stage m_stage;
    data::Vec3 m_grad;
    double m_round_starting_objective;
    bool m_terminate;
    // Gradient Finding
    unsigned short m_current_dim;
    // Line Search
    std::pair<data::Vec3, double> m_round_point;
    double m_delta;
    double m_last_dv;
};

void export_linesearch(py::module& m);
}} // namespace pgop::optimize
