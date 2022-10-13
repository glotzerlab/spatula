#include <algorithm>
#include <cmath>
#include <limits>

#include <pybind11/pybind11.h>

#include "Optimize.h"

namespace pgop { namespace optimize {
Optimizer::Optimizer(const std::vector<double>& min_bounds, const std::vector<double>& max_bounds)
    : m_min_bounds(min_bounds), m_max_bounds(max_bounds), m_point(), m_objective(),
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
}

void Optimizer::clip_point(std::vector<double>& point)
{
    for (size_t i {0}; i < point.size(); ++i) {
        point[i] = std::clamp(point[i], m_min_bounds[i], m_max_bounds[i]);
    }
}

std::vector<double> PyOptimizer::next_point()
{
    PYBIND11_OVERRIDE_PURE(std::vector<double>, Optimizer, next_point);
}

void PyOptimizer::record_objective(double objective)
{
    PYBIND11_OVERRIDE_PURE(void, Optimizer, record_objective, objective);
}
bool PyOptimizer::terminate() const
{
    PYBIND11_OVERRIDE_PURE(bool, Optimizer, terminate);
}
std::pair<std::vector<double>, double> PyOptimizer::get_optimum() const
{
    using pair_ = std::pair<std::vector<double>, double>;
    PYBIND11_OVERRIDE_PURE_NAME(pair_, Optimizer, "optimum", get_optimum);
}

std::unique_ptr<Optimizer> PyOptimizer::clone() const
{
    return std::make_unique<PyOptimizer>(*this);
}
}} // namespace pgop::optimize
