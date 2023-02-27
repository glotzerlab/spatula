#pragma once

#include <random>

#include <pybind11/pybind11.h>

#include "Optimize.h"

namespace pgop { namespace optimize {

class StepGradientDescent : public Optimizer {
    public:
    StepGradientDescent(const data::Vec3& initial_point,
                        unsigned int max_iter,
                        double initial_jump,
                        double learning_rate,
                        double tol);
    ~StepGradientDescent() override = default;
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
    data::Vec3 computeNewRotation() const;
    void searchAlongGradient();

    // Hyperparameters
    unsigned int m_max_iter;
    double m_initial_jump;
    double m_learning_rate;
    double m_tol;
    // State variables
    // General
    Stage m_stage;
    double m_dim_starting_objective;
    bool m_terminate;
    // Gradient Descent
    unsigned short m_current_dim;
    double m_last_objective;
    double m_delta;
};

void export_step_gradient_descent(py::module& m);
}} // namespace pgop::optimize
