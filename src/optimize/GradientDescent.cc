#include <cmath>
#include <numeric>

#include <pybind11/stl.h>

#include "GradientDescent.h"

namespace pgop { namespace optimize {
GradientDescent::GradientDescent(const std::vector<double>& min_bounds,
                                 const std::vector<double>& max_bounds,
                                 const std::vector<double>& initial_point,
                                 double alpha,
                                 double max_move_size,
                                 double tol,
                                 unsigned int n_rounds)
    : Optimizer(min_bounds, max_bounds),
      m_best_point(initial_point, std::numeric_limits<double>::max()), m_last_point(),
      m_current_dim(0), m_current_opt_count(0), m_n_rounds(0), m_n_rounds_max(n_rounds),
      m_alpha(alpha), m_max_move_size(max_move_size), m_tol(tol), m_delta()
{
    m_point = initial_point;
}

void GradientDescent::internal_next_point()
{
    if (m_count == 0) {
        return;
    }
    if (m_count == 1) {
        m_current_opt_count += 1;
        m_last_point.first = m_point;
        m_last_point.second = m_objective;
        m_point[m_current_dim] += getInitialDelta();
        return;
    }
    m_current_opt_count += 1;
    const double obj_change = m_objective - m_last_point.second;
    // Don't do more than 10 iterations on the same dimension in a row.
    if (std::abs(obj_change) < m_tol || m_current_opt_count > 10) {
        m_current_opt_count = 0;
        m_last_point.first = m_point;
        m_last_point.second = m_objective;
        m_current_dim = (m_current_dim + 1) % m_min_bounds.size();
        if (m_current_dim == 0) {
            m_n_rounds += 1;
        }
        m_point[m_current_dim] += getInitialDelta();
    } else {
        const double delta_x = m_last_point.first[m_current_dim] - m_point[m_current_dim];
        m_last_point.first = m_point;
        m_last_point.second = m_objective;
        // we multiply by the negative gradient so we can add.
        m_point[m_current_dim] += m_alpha * obj_change / delta_x;
    }
}

bool GradientDescent::terminate() const
{
    return m_n_rounds >= m_n_rounds_max;
}

std::unique_ptr<Optimizer> GradientDescent::clone() const
{
    return std::make_unique<GradientDescent>(*this);
}

unsigned int GradientDescent::getNRounds() const
{
    return m_n_rounds;
}
unsigned int GradientDescent::getCurrentDim() const
{
    return m_current_dim;
}

double GradientDescent::getAlpha() const
{
    return m_alpha;
}
void GradientDescent::setAlpha(double alpha)
{
    m_alpha = alpha;
}

double GradientDescent::getMaxMoveSize() const
{
    return m_max_move_size;
}
void GradientDescent::setMaxMoveSize(double max_move_size)
{
    m_max_move_size = max_move_size;
}

double GradientDescent::getTol() const
{
    return m_tol;
}
void GradientDescent::setTol(double tol)
{
    m_tol = tol;
}

unsigned int GradientDescent::getNRoundsMax() const
{
    return m_n_rounds_max;
}
void GradientDescent::setNRoundsMax(unsigned int n_rounds)
{
    m_n_rounds_max = n_rounds;
}

double GradientDescent::getInitialDelta() const
{
    const double range = m_max_bounds[m_current_dim] - m_min_bounds[m_current_dim];
    if (std::isnan(range) || range == std::numeric_limits<double>::max()) {
        return 0.02 * m_point[m_current_dim];
    }
    return 0.02 * range;
}

void export_gradient_descent(py::module& m)
{
    py::class_<GradientDescent, Optimizer, std::shared_ptr<GradientDescent>>(m, "GradientDescent")
        .def(py::init<const std::vector<double>&,
                      const std::vector<double>&,
                      const std::vector<double>&,
                      double,
                      double,
                      double,
                      unsigned int>())
        .def_property_readonly("n_rounds", &GradientDescent::getNRounds)
        .def_property_readonly("current_dim", &GradientDescent::getCurrentDim)
        .def_property("alpha", &GradientDescent::getAlpha, &GradientDescent::setAlpha)
        .def_property("max_move_size",
                      &GradientDescent::getMaxMoveSize,
                      &GradientDescent::setMaxMoveSize)
        .def_property("tol", &GradientDescent::getTol, &GradientDescent::setTol)
        .def_property("n_rounds_max",
                      &GradientDescent::getNRoundsMax,
                      &GradientDescent::setNRoundsMax);
}
}} // namespace pgop::optimize
