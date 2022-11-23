#include "LocalFIRE.h"

namespace pgop { namespace optimize {

LocalFIRE::LocalFIRE(const data::Quaternion& initial_point, unsigned int max_iter)
    : Optimizer(), m_max_iter(max_iter), m_opt_cnt(0), m_stage(LocalFIRE::Stage::GRADIENT),
      m_current_dim(0), m_opt_point(), m_grad(),
      m_axes({1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}), m_beta(0.5), m_delta(0)
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
        m_opt_cnt = 0;
        m_delta = 0;
        m_stage = LocalFIRE::Stage::SEARCH;
    } else if (m_stage == LocalFIRE::Stage::SEARCH) {
        /* py::print("m_opt_cnt =", m_opt_cnt, "end"_a=", "); */
        m_stage = LocalFIRE::Stage::GRADIENT;
        m_current_dim = 0;
        m_opt_point = m_best_point;
    }
}

void LocalFIRE::findGradient()
{
    if (m_current_dim == 0) {
        m_point = data::Quaternion(std::get<0>(m_axes), 0.017) * m_opt_point.first;
    } else if (m_current_dim == 1) {
        m_grad[0] = m_objective - m_opt_point.second;
        m_point = data::Quaternion(std::get<1>(m_axes), 0.017) * m_opt_point.first;
    } else if (m_current_dim == 2) {
        m_grad[1] = m_objective - m_opt_point.second;
        m_point = data::Quaternion(std::get<2>(m_axes), 0.017) * m_opt_point.first;
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
           * data::Quaternion(std::get<0>(m_axes), m_grad[0] * m_delta) * m_point;
}

void LocalFIRE::searchAlongGradient()
{
    ++m_opt_cnt;
    const double objective_change = m_opt_point.second - m_objective;
    if (m_delta == 0) {
        m_delta = 0.1;
        // Both ensure that we are improving and at an appreciable rate.
    } else if (objective_change > 0) {
        /* m_opt_point.first = m_point; */
        /* m_opt_point.second = m_objective; */
        const double grad = objective_change / m_delta;
        m_delta = (m_delta * m_beta + (1 - m_beta) * grad);
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
        .def(py::init<const data::Quaternion&, unsigned int>());
}
}} // namespace pgop::optimize
