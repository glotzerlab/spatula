#include <cmath>
#include <limits>

#include "BruteForce.h"

namespace pgop { namespace optimize {
BruteForce::BruteForce(const std::vector<std::vector<double>>& points,
                       const std::vector<double>& min_bounds,
                       const std::vector<double>& max_bounds)
    : Optimizer(min_bounds, max_bounds), m_points(points), m_best_point(min_bounds.size(), NAN),
      m_best_objective(std::numeric_limits<double>::infinity())
{
}

void BruteForce::internal_next_point()
{
    m_point = m_points[std::min(m_points.size(), static_cast<size_t>(m_count))];
}

bool BruteForce::terminate() const
{
    return m_count >= m_points.size();
}

std::unique_ptr<Optimizer> BruteForce::clone() const
{
    return std::make_unique<BruteForce>(*this);
}
}} // end namespace pgop::optimize
