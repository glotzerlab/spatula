#include "LineSearch.h"

namespace pgop { namespace optimize {

LineSearch::LineSearch(const data::Vec3& initial_point,
                       unsigned int max_iter,
                       double initial_jump,
                       double learning_rate,
                       double tol)
    : Optimizer(), m_max_iter(max_iter), m_learning_rate(learning_rate),
      m_initial_jump(initial_jump), m_tol(tol), m_stage(LineSearch::Stage::INITIALIZE), m_grad(),
      m_terminate(false), m_current_dim(0), m_round_point({}, 0.0), m_delta(0.05), m_last_dv(0.0)
{
    m_point = initial_point;
}

bool LineSearch::terminate() const
{
    bool term = m_terminate || m_count > m_max_iter;
    if (term) { }
    return term;
}

std::unique_ptr<Optimizer> LineSearch::clone() const
{
    return std::make_unique<LineSearch>(*this);
}

void LineSearch::internal_next_point()
{
    if (m_stage == LineSearch::Stage::INITIALIZE) {
        initialize();
    }
    if (m_stage == LineSearch::Stage::GRADIENT) {
        findGradient();
    } else if (m_stage == LineSearch::Stage::SEARCH) {
        searchAlongGradient();
        if (m_stage == LineSearch::Stage::GRADIENT) {
            findGradient();
        }
    }
}

void LineSearch::step()
{
    if (m_stage == LineSearch::Stage::INITIALIZE) {
        m_round_starting_objective = m_best_point.second;
        m_round_point.second = m_best_point.second;
        m_stage = LineSearch::Stage::GRADIENT;
    } else if (m_stage == LineSearch::Stage::GRADIENT) {
        m_stage = LineSearch::Stage::SEARCH;
    } else if (m_stage == LineSearch::Stage::SEARCH) {
        m_stage = LineSearch::Stage::GRADIENT;
        m_current_dim = 0;
        m_point.z -= m_initial_jump;
        m_terminate = std::abs(m_best_point.second - m_round_starting_objective) < m_tol;
        m_round_starting_objective = m_best_point.second;
    }
}

void LineSearch::initialize()
{
    if (m_best_point.second == std::numeric_limits<double>::max()) {
        return;
    }
    step();
}

void LineSearch::findGradient()
{
    if (m_current_dim == 0) {
        m_point.x += m_initial_jump;
        // a hack to not need another float. m_delta stores the initial objective before finding any
        // dimension's gradient.
        m_delta = m_objective;
        m_round_point.second = m_objective;
    } else if (m_current_dim == 1) {
        m_grad[0] = (m_objective - m_round_point.second) / m_initial_jump;
        m_point.x -= m_initial_jump;
        m_point.y += m_initial_jump;
        m_round_point.second = m_objective;
    } else if (m_current_dim == 2) {
        m_grad[1] = (m_objective - m_round_point.second) / m_initial_jump;
        m_point.y -= m_initial_jump;
        m_point.z += m_initial_jump;
        m_round_point.second = m_objective;
    } else if (m_current_dim == 3) {
        m_grad[2] = (m_objective - m_round_point.second) / m_initial_jump;
        m_point.z -= m_initial_jump;
        m_round_point.first = m_point;
        // restore m_delta.
        m_round_point.second = m_delta;
        m_delta = 0.05;
        m_last_dv = m_delta;
        m_point = m_round_point.first - (m_grad * m_delta);
        step();
    }
    ++m_current_dim;
}

void LineSearch::searchAlongGradient()
{
    const double objective_change = m_objective - m_round_point.second;
    m_round_point.second = m_objective;
    if (std::abs(objective_change) < 1e-4) {
        step();
        return;
    }
    const double grad = objective_change / m_last_dv;
    m_last_dv = -m_learning_rate * grad;
    m_delta += m_last_dv;
    m_point = m_round_point.first - (m_grad * m_delta);
}

void export_linesearch(py::module& m)
{
    py::class_<LineSearch, Optimizer, std::shared_ptr<LineSearch>>(m, "LineSearch")
        .def(py::init<const data::Vec3&, unsigned int, double, double, double>());
}
}} // namespace pgop::optimize
