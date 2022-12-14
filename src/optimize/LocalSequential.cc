#include "LocalSequential.h"

namespace pgop { namespace optimize {

LocalSequential::LocalSequential(const data::Quaternion& initial_point,
                                 unsigned int max_iter,
                                 double initial_jump)
    : Optimizer(), m_max_iter(max_iter), m_stage(LocalSequential::Stage::GRADIENT),
      m_current_dim(0), m_opt_point(), m_axes {data::Vec3 {1.0, 0.0, 0.0},
                                               data::Vec3 {0.0, 1.0, 0.0},
                                               data::Vec3 {0.0, 0.0, 1.0}},
      m_grad(), m_beta(0.5), m_delta(0), m_initial_jump(initial_jump)
{
    m_point = initial_point;
}

bool LocalSequential::terminate() const
{
    return m_count > m_max_iter;
}

std::unique_ptr<Optimizer> LocalSequential::clone() const
{
    return std::make_unique<LocalSequential>(*this);
}

void LocalSequential::internal_next_point()
{
    if (m_best_point.second == std::numeric_limits<double>::max()) {
        return;
    }
    if (m_stage == LocalSequential::Stage::GRADIENT) {
        findGradient();
    }
    if (m_stage == LocalSequential::Stage::SEARCH) {
        searchAlongGradient();
        if (m_stage == LocalSequential::Stage::GRADIENT) {
            findGradient();
        }
    }
}

void LocalSequential::step()
{
    /* using namespace py::literals; */
    if (m_stage == LocalSequential::Stage::GRADIENT) {
        m_delta = 0;
        m_stage = LocalSequential::Stage::SEARCH;
    } else if (m_stage == LocalSequential::Stage::SEARCH) {
        m_stage = LocalSequential::Stage::GRADIENT;
        m_current_dim = (m_current_dim + 1) % 3;
        m_opt_point = m_best_point;
    }
}

void LocalSequential::findGradient()
{
    m_angle = m_initial_jump;
    m_point = data::Quaternion(m_axes[m_current_dim], m_angle) * m_opt_point.first;
    step();
}

data::Quaternion LocalSequential::computeNewRotation() const
{
    return data::Quaternion(m_axes[m_current_dim], m_angle) * m_opt_point.first;
}

void LocalSequential::searchAlongGradient()
{
    const double objective_change = m_objective - m_opt_point.second;
    if (m_angle == m_initial_jump) {
        m_grad = objective_change / m_angle;
        m_angle = (1 - m_beta) * m_grad;
    } else if (objective_change < 0) {
        // Only update objective because we are using gradient descent so the effect of a slight
        // change in the angle of rotation must be conserved which it isn't if we update points.
        m_opt_point.second = m_objective;
        m_grad = objective_change / m_angle;
        m_angle = (m_beta * m_angle) - (1 - m_beta) * m_grad;
        // Went uphill readjust direction.
    } else {
        step();
        return;
    }
    m_point = computeNewRotation();
}

void export_local_seq(py::module& m)
{
    py::class_<LocalSequential, Optimizer, std::shared_ptr<LocalSequential>>(m, "LocalSequential")
        .def(py::init<const data::Quaternion&, unsigned int, double>());
}
}} // namespace pgop::optimize
