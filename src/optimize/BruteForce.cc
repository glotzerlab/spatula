#include <cmath>
#include <limits>

#include "BruteForce.h"

namespace pgop { namespace optimize {
BruteForce::BruteForce(const std::vector<std::vector<double>>& points,
                       const std::vector<double>& min_bounds,
                       const std::vector<double>& max_bounds)
    : Optimizer(min_bounds, max_bounds), m_points(points), m_cnt(0),
      m_best_point(min_bounds.size(), NAN),
      m_best_objective(std::numeric_limits<double>::infinity())
{
}

std::vector<double> BruteForce::next_point()
{
    m_need_objective = true;
    m_point = m_points[std::min(m_points.size(), m_cnt)];
    ++m_cnt;
    return m_point;
}

std::pair<std::vector<double>, double> BruteForce::get_optimum() const
{
    return std::make_pair(m_best_point, m_best_objective);
}

void BruteForce::record_objective(double objective)
{
    if (!m_need_objective) {
        throw std::runtime_error("Must get new point before recording objective.");
    }
    m_need_objective = false;
    m_objective = objective;
    if (objective < m_best_objective) {
        m_best_objective = objective;
        m_best_point = m_point;
    }
}

bool BruteForce::terminate() const
{
    return m_cnt >= m_points.size();
}

std::unique_ptr<Optimizer> BruteForce::clone() const
{
    return std::make_unique<BruteForce>(*this);
}
}} // end namespace pgop::optimize
