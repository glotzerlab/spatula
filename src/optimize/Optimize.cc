#include <algorithm>
#include <cmath>
#include <limits>

#include <pybind11/pybind11.h>

#include "Optimize.h"

namespace pgop { namespace optimize {
Optimizer::Optimizer(const std::vector<double>& min_bounds, const std::vector<double>& max_bounds)
    : m_min_bounds(min_bounds), m_max_bounds(max_bounds), m_point(), m_objective(), m_count(0),
      m_need_objective(false)
{
}

void Optimizer::record_objective(double objective)
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

std::pair<std::vector<double>, double> Optimizer::get_optimum() const
{
    return m_best_point;
}

void Optimizer::clip_point(std::vector<double>& point)
{
    for (size_t i {0}; i < point.size(); ++i) {
        point[i] = std::clamp(point[i], m_min_bounds[i], m_max_bounds[i]);
    }
}

std::vector<double> Optimizer::next_point()
{
    if (m_need_objective) {
        throw std::runtime_error("Must record objective for new point first.");
    }
    internal_next_point();
    clip_point(m_point);
    ++m_count;
    m_need_objective = true;
    return m_point;
}

unsigned int Optimizer::getCount() const
{
    return m_count;
}
const std::vector<double>& Optimizer::getMinBounds() const
{
    return m_min_bounds;
}
const std::vector<double>& Optimizer::getMaxBounds() const
{
    return m_max_bounds;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
void Optimizer::specialize(unsigned int particle_index) { }
#pragma GCC diagnostic pop

void PyOptimizer::internal_next_point()
{
    PYBIND11_OVERRIDE_PURE(void, Optimizer, internal_next_point);
}

void PyOptimizer::record_objective(double objective)
{
    PYBIND11_OVERRIDE(void, Optimizer, record_objective, objective);
}
bool PyOptimizer::terminate() const
{
    PYBIND11_OVERRIDE_PURE(bool, Optimizer, terminate);
}

std::unique_ptr<Optimizer> PyOptimizer::clone() const
{
    return std::make_unique<PyOptimizer>(*this);
}
}} // namespace pgop::optimize
