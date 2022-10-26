#include <cmath>
#include <numeric>

#include <pybind11/stl.h>

#include "LocalBinaryOptimizer.h"

namespace pgop { namespace optimize {
LocalBinaryOptimizer::LocalBinaryOptimizer(const std::vector<double>& min_bounds,
                                           const std::vector<double>& max_bounds,
                                           const std::vector<double>& initial_point,
                                           double max_move_size,
                                           unsigned int iter_max)
    : Optimizer(min_bounds, max_bounds), m_last_point(),
      m_stage(LocalBinaryOptimizer::Stage::JACOBIAN), m_terminate(false), m_current_dim(0),
      m_grad(), m_max_move_size(max_move_size), m_opt_steps(0), m_scale(1), m_opt_objectives {0, 0},
      m_iter(0), m_iter_max(iter_max)
{
    m_point = initial_point;
}

void LocalBinaryOptimizer::internal_next_point()
{
    if (m_count == 0) {
        return;
    }
    if (m_count == 1) {
        m_last_point.first = m_point;
        m_last_point.second = m_objective;
    }
    if (m_stage == LocalBinaryOptimizer::Stage::JACOBIAN) {
        findJacobian();
    }
    // We may have finished finding the Jacobian. If so begin minimization.
    if (m_stage == LocalBinaryOptimizer::Stage::MINIMIZE) {
        findContourMin();
        // Similarly we may have finished minimization. If so we should begin finding the new
        // Jacobian.
        if (m_stage == LocalBinaryOptimizer::Stage::JACOBIAN) {
            findJacobian();
        }
    }
}

bool LocalBinaryOptimizer::terminate() const
{
    return m_terminate || m_iter >= m_iter_max;
}

std::unique_ptr<Optimizer> LocalBinaryOptimizer::clone() const
{
    return std::make_unique<LocalBinaryOptimizer>(*this);
}

unsigned int LocalBinaryOptimizer::getIter() const
{
    return m_iter;
}
unsigned int LocalBinaryOptimizer::getCurrentDim() const
{
    return m_current_dim;
}

unsigned int LocalBinaryOptimizer::getIterMax() const
{
    return m_iter_max;
}
void LocalBinaryOptimizer::setIterMax(unsigned int iter_max)
{
    m_iter_max = iter_max;
}

void LocalBinaryOptimizer::step()
{
    if (m_stage == LocalBinaryOptimizer::Stage::JACOBIAN) {
        m_opt_steps = 0;
        m_scale = 1.0;
        m_stage = LocalBinaryOptimizer::Stage::MINIMIZE;
    } else {
        ++m_iter;
        m_current_dim = 0;
        m_grad.clear();
        m_last_point = m_best_point;
        m_stage = LocalBinaryOptimizer::Stage::JACOBIAN;
    }
}

void LocalBinaryOptimizer::findJacobian()
{
    // Initiate computing the Jacobian
    if (m_current_dim == 0) {
        // Modify next dimension to determine the Jacobian.
        m_point[m_current_dim] += getInitialDelta();
        ++m_current_dim;
        return;
    }
    m_grad.emplace_back((m_objective - m_last_point.second)
                        / (m_point[m_current_dim - 1] - m_last_point.first[m_current_dim - 1]));
    // Restore m_point to starting position.
    m_point[m_current_dim - 1] = m_last_point.first[m_current_dim - 1];
    // Switch to minimization.
    if (m_grad.size() == m_max_bounds.size()) {
        normalizeGradient();
        step();
        return;
    }
    // Modify next dimension to determine the Jacobian.
    m_point[m_current_dim] += getInitialDelta();
    ++m_current_dim;
}

double LocalBinaryOptimizer::getInitialDelta() const
{
    const double range = m_max_bounds[m_current_dim] - m_min_bounds[m_current_dim];
    if (std::isnan(range) || range == std::numeric_limits<double>::max()) {
        return std::min(std::abs(0.02 * m_point[m_current_dim]), 0.5 * m_max_move_size);
    }
    return std::min(0.02 * range, 0.5 * m_max_move_size);
}

void LocalBinaryOptimizer::normalizeGradient()
{
    const double norm = std::transform_reduce(m_grad.cbegin(), m_grad.cend(), m_grad.cbegin(), 0.0);
    const double scale_dim = m_max_move_size / std::sqrt(norm);
    for (auto& x : m_grad) {
        x *= scale_dim;
    }
}

void LocalBinaryOptimizer::findContourMin()
{
    // Optimization termination conditions
    if (m_opt_steps > 7 && m_best_point.second < m_last_point.second) {
        m_point = m_best_point.first;
        step();
        return;
    }
    // Cannot optimize in this direction seemingly just return.
    if (m_opt_steps > 11) {
        // Terminate if the current search start is the best point. It could be that one of the
        // points tried to compute the Jacobian was the best. If so, we will continue from that
        // point.
        m_terminate = m_last_point.second == m_best_point.second;
        if (!m_terminate) {
            step();
        }
        return;
    }
    if (m_opt_steps == 0) {
        m_opt_objectives.first = m_last_point.second;
    } else if (m_opt_steps == 1) {
        m_opt_objectives.second = m_objective;
        m_scale = 0.5;
    } else {
        const bool use_left = m_opt_objectives.first < m_opt_objectives.second;
        if (use_left) {
            m_opt_objectives.second = m_objective;
        } else {
            m_opt_objectives.first = m_objective;
        }
        const double scale_change_sign = use_left ? -1 : 1;
        m_scale += scale_change_sign * std::pow(0.5, m_opt_steps);
    }
    for (size_t i {0}; i < m_grad.size(); ++i) {
        m_point[i] = m_scale * m_grad[i] + m_last_point.first[i];
    }
    ++m_opt_steps;
}

void export_local_binary(py::module& m)
{
    py::class_<LocalBinaryOptimizer, Optimizer, std::shared_ptr<LocalBinaryOptimizer>>(
        m,
        "LocalBinaryOptimizer")
        .def(py::init<const std::vector<double>&,
                      const std::vector<double>&,
                      const std::vector<double>&,
                      double,
                      unsigned int>())
        .def_property_readonly("iter", &LocalBinaryOptimizer::getIter)
        .def_property_readonly("current_dim", &LocalBinaryOptimizer::getCurrentDim)
        .def_property("iter_max",
                      &LocalBinaryOptimizer::getIterMax,
                      &LocalBinaryOptimizer::setIterMax);
}
}} // namespace pgop::optimize
