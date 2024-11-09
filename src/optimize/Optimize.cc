#include <algorithm>
#include <cmath>
#include <limits>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "Optimize.h"

namespace pgop { namespace optimize {
Optimizer::Optimizer()
    : m_point(), m_objective(), m_best_point({0.0, 0.0, 0.0}, std::numeric_limits<double>::max()),
      m_count(0), m_need_objective(false)
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

std::pair<data::Vec3, double> Optimizer::get_optimum() const
{
    return m_best_point;
}

data::Vec3 Optimizer::next_point()
{
    if (m_need_objective) {
        throw std::runtime_error("Must record objective for new point first.");
    }
    internal_next_point();
    ++m_count;
    m_need_objective = true;
    return m_point;
}

unsigned int Optimizer::getCount() const
{
    return m_count;
}

void PyOptimizer::internal_next_point()
{
    NB_OVERRIDE_PURE(internal_next_point);
}

void PyOptimizer::record_objective(double objective)
{
    NB_OVERRIDE(record_objective, objective);
}
bool PyOptimizer::terminate() const
{
    NB_OVERRIDE_PURE(terminate);
}

std::unique_ptr<Optimizer> PyOptimizer::clone() const
{
    return std::make_unique<PyOptimizer>(*this);
}

void export_base_optimize(nb::module_& m)
{
    nb::class_<Optimizer, PyOptimizer>(m, "Optimizer")
        .def("record_objective", &Optimizer::record_objective)
        .def_prop_ro("terminate", &Optimizer::terminate)
        .def_prop_ro("count", &Optimizer::getCount);
}
}} // namespace pgop::optimize
