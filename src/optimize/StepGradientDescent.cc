// Copyright (c) 2010-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include "StepGradientDescent.h"

namespace spatula { namespace optimize {

StepGradientDescent::StepGradientDescent(const data::Vec3& initial_point,
                                         unsigned int max_iter,
                                         double initial_jump,
                                         double learning_rate,
                                         double tol)
    : Optimizer(), m_max_iter(max_iter), m_initial_jump(initial_jump),
      m_learning_rate(learning_rate), m_tol(tol), m_stage(StepGradientDescent::Stage::INITIALIZE),
      m_dim_starting_objective(0.0), m_terminate(false), m_current_dim(0), m_last_objective(0.0),
      m_delta(0.0)
{
    m_point = initial_point;
}

bool StepGradientDescent::terminate() const
{
    const bool term = m_terminate || m_count > m_max_iter;
    if (term) { }
    return term;
}

std::unique_ptr<Optimizer> StepGradientDescent::clone() const
{
    return std::make_unique<StepGradientDescent>(*this);
}

void StepGradientDescent::internal_next_point()
{
    if (m_stage == StepGradientDescent::Stage::INITIALIZE) {
        initialize();
    }
    if (m_stage == StepGradientDescent::Stage::GRADIENT) {
        findGradient();
    }
    if (m_stage == StepGradientDescent::Stage::SEARCH) {
        searchAlongGradient();
        if (m_stage == StepGradientDescent::Stage::GRADIENT) {
            findGradient();
        }
    }
}

void StepGradientDescent::initialize()
{
    if (m_best_point.second == std::numeric_limits<double>::max()) {
        return;
    }
    step();
}

void StepGradientDescent::step()
{
    if (m_stage == StepGradientDescent::Stage::INITIALIZE) {
        m_last_objective = m_best_point.second;
        m_dim_starting_objective = m_best_point.second;
        m_stage = StepGradientDescent::Stage::GRADIENT;
    }
    if (m_stage == StepGradientDescent::Stage::GRADIENT) {
        m_stage = StepGradientDescent::Stage::SEARCH;
    } else if (m_stage == StepGradientDescent::Stage::SEARCH) {
        m_stage = StepGradientDescent::Stage::GRADIENT;
        m_current_dim = (m_current_dim + 1) % 3;
        if (m_current_dim == 0) {
            m_terminate = (m_dim_starting_objective - m_best_point.second) < m_tol;
        }
        m_dim_starting_objective = m_best_point.second;
    }
}

void StepGradientDescent::findGradient()
{
    m_point = m_best_point.first;
    m_point[m_current_dim] -= m_initial_jump;
    m_delta = m_initial_jump;
    step();
}

void StepGradientDescent::searchAlongGradient()
{
    const double objective_change = m_objective - m_last_objective;
    m_last_objective = m_objective;
    if (std::abs(objective_change) < m_tol) {
        step();
        return;
    }
    const double grad = -objective_change / (m_delta);
    m_delta = m_learning_rate * grad;
    m_point[m_current_dim] -= m_delta;
}

void export_step_gradient_descent(py::module& m)
{
    py::class_<StepGradientDescent, Optimizer, std::shared_ptr<StepGradientDescent>>(
        m,
        "StepGradientDescent")
        .def(py::init<const data::Vec3&, unsigned int, double, double, double>());
}
}} // namespace spatula::optimize
