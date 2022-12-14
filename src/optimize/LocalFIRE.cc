#include "LocalFIRE.h"

namespace pgop { namespace optimize {

LocalFIRE::LocalFIRE(const data::Quaternion& initial_point,
                     unsigned int max_iter,
                     double initial_jump)
    : Optimizer(), m_max_iter(max_iter), m_stage(LocalFIRE::Stage::GRADIENT), m_current_dim(0),
      m_opt_point(), m_grad(), m_axes({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}),
      m_beta(0.5), m_delta(0), m_initial_jump(initial_jump)
{
    m_point = initial_point;
}

bool LocalFIRE::terminate() const
{
    return m_count > m_max_iter;
}

std::unique_ptr<Optimizer> LocalFIRE::clone() const
{
    return std::make_unique<LocalFIRE>(*this);
}

void LocalFIRE::internal_next_point()
{
    if (m_best_point.second == std::numeric_limits<double>::max()) {
        return;
    }
    if (m_stage == LocalFIRE::Stage::GRADIENT) {
        findGradient();
    }
    if (m_stage == LocalFIRE::Stage::SEARCH) {
        searchAlongGradient();
        if (m_stage == LocalFIRE::Stage::GRADIENT) {
            findGradient();
        }
    }
}

void LocalFIRE::step()
{
    /* using namespace py::literals; */
    if (m_stage == LocalFIRE::Stage::GRADIENT) {
        m_delta = 0;
        m_stage = LocalFIRE::Stage::SEARCH;
    } else if (m_stage == LocalFIRE::Stage::SEARCH) {
        m_stage = LocalFIRE::Stage::GRADIENT;
        m_current_dim = 0;
        m_opt_point = m_best_point;
    }
}

void LocalFIRE::findGradient()
{
    if (m_current_dim == 0) {
        m_point = data::Quaternion(std::get<0>(m_axes), m_initial_jump) * m_opt_point.first;
    } else if (m_current_dim == 1) {
        m_grad[0] = m_objective - m_opt_point.second;
        m_point = data::Quaternion(std::get<1>(m_axes), m_initial_jump) * m_opt_point.first;
    } else if (m_current_dim == 2) {
        m_grad[1] = m_objective - m_opt_point.second;
        m_point = data::Quaternion(std::get<2>(m_axes), m_initial_jump) * m_opt_point.first;
    } else if (m_current_dim == 3) {
        m_grad[2] = m_objective - m_opt_point.second;
        step();
        return;
    }
    ++m_current_dim;
}

data::Quaternion LocalFIRE::computeNewRotation() const
{
    return data::Quaternion(std::get<2>(m_axes), m_grad[2] * m_delta)
           * data::Quaternion(std::get<1>(m_axes), m_grad[1] * m_delta)
           * data::Quaternion(std::get<0>(m_axes), m_grad[0] * m_delta) * m_opt_point.first;
}

void LocalFIRE::searchAlongGradient()
{
    const double objective_change = m_objective - m_opt_point.second;
    if (m_delta == 0) {
        m_delta = 0.05;
        // Ensure that we are improving.
    } else if (objective_change < 0) {
        m_opt_point.second = m_objective;
        const double grad = objective_change / m_delta;
        m_delta = (m_beta * m_delta) - (1 - m_beta) * grad;
        // Went uphill readjust direction.
    } else {
        step();
        return;
    }
    m_point = computeNewRotation();
}

void export_localfire(py::module& m)
{
    py::class_<LocalFIRE, Optimizer, std::shared_ptr<LocalFIRE>>(m, "LocalFIRE")
        .def(py::init<const data::Quaternion&, unsigned int, double>());
}
}} // namespace pgop::optimize
