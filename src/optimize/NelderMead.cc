#include <algorithm>
#include <cmath>
#include <limits>

#include "NelderMead.h"

namespace pgop { namespace optimize {
NelderMeadParams::NelderMeadParams(double alpha_, double gamma_, double rho_, double sigma_)
    : alpha(alpha_), gamma(gamma_), rho(rho_), sigma(sigma_)
{
}

RollingStd::RollingStd() : m_mean {0}, m_var {0}, m_n {0} { }

RollingStd::RollingStd(const std::vector<double>& values)
    : m_mean {0}, m_var {0}, m_n {static_cast<double>(values.size())}
{
    double sq_mean {0};
    for (const auto& v : values) {
        m_mean += v;
        sq_mean += v * v;
    }
    m_var = (sq_mean - (m_mean * m_mean / m_n)) / m_n;
    m_mean /= m_n;
}

void RollingStd::update(double new_value, double old_value)
{
    double old_mean = m_mean;
    double delta = new_value - old_value;
    m_mean += delta / m_n;
    m_var += delta * (new_value - m_mean + old_value - old_mean) / m_n;
}

double RollingStd::std() const
{
    return std::sqrt(m_var);
}

double RollingStd::mean() const
{
    return m_mean;
}

double compute_distance(const std::vector<double>& a, const std::vector<double>& b)
{
    double dist {0};
    for (size_t i {0}; i < a.size(); ++i) {
        const double diff = a[i] - b[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

OrderedSimplex::OrderedSimplex(unsigned int dim)
    : m_dim(dim), m_points(), m_rolling_std(), m_min_dist(std::numeric_limits<double>::infinity())
{
}

void OrderedSimplex::add(const std::vector<double>& point, double objective)
{
    if (m_points.size() < m_dim + 1) {
        m_points.emplace_back(std::make_pair(point, objective));
        if (m_points.size() == m_dim + 1) {
            complete_initialization();
        }
        return;
    }
    // Order in place
    std::pair<std::vector<double>, double> tmp_point;
    size_t nth_lowest {m_dim + 1};
    for (size_t i {0}; i < m_points.size(); ++i) {
        if (objective < m_points[i].second) {
            tmp_point = m_points[i];
            nth_lowest = i;
            break;
        }
    }
    // Is the highest value or tied for highest
    if (nth_lowest == m_dim + 1) {
        return;
    }
    // Update min_dist before adding new points since we do not know what index
    // we will add to, but we do know that the last point is the one to be removed.
    update_min_distance(point);
    m_points[nth_lowest] = std::make_pair(point, objective);

    for (size_t i {nth_lowest + 1}; i < m_points.size(); ++i) {
        auto tmp_point_2 = m_points[i];
        m_points[i] = tmp_point;
        tmp_point = tmp_point_2;
    }
    // Update statistics
    m_rolling_std.update(objective, tmp_point.second);
}

size_t OrderedSimplex::size() const
{
    return m_points.size();
}

void OrderedSimplex::complete_initialization()
{
    // Order simplex
    std::sort(m_points.begin(), m_points.end(), [](auto& x, auto& y) {
        return x.second < y.second;
    });
    // Compute statistics on simplex for determining termination
    std::vector<double> objectives(m_dim + 1);
    std::transform(m_points.begin(), m_points.end(), objectives.begin(), [](const auto& point) {
        return point.second;
    });
    m_rolling_std = RollingStd(objectives);
    // Compute minimum distance in simplex
    for (size_t i {0}; i < m_dim + 1; ++i) {
        for (size_t j {i + 1}; j < m_dim + 1; ++j) {
            auto dist = compute_distance(m_points[i].first, m_points[j].first);
            if (dist < m_min_dist) {
                m_min_dist = dist;
            }
        }
    }
}

const std::vector<double>& OrderedSimplex::get_point(size_t index) const
{
    return m_points[index].first;
}

double OrderedSimplex::get_objective(size_t index) const
{
    return m_points[index].second;
}

double OrderedSimplex::get_objective_std() const
{
    return m_rolling_std.std();
}

double OrderedSimplex::get_objective_mean() const
{
    return m_rolling_std.mean();
}

double OrderedSimplex::get_min_dist() const
{
    return m_min_dist;
}

const std::pair<std::vector<double>, double>& OrderedSimplex::operator[](size_t index) const
{
    return m_points[index];
}

std::vector<double> OrderedSimplex::compute_centroid() const
{
    auto centroid = std::vector<double>(m_dim, 0);
    for (size_t i {0}; i < m_dim; ++i) {
        for (size_t j {0}; j < m_dim; ++j) {
            centroid[j] += m_points[i].first[j];
        }
    }
    for (size_t i {0}; i < m_dim; ++i) {
        centroid[i] /= static_cast<double>(m_dim);
    }
    return centroid;
}

void OrderedSimplex::update_min_distance(const std::vector<double>& new_point)
{
    for (size_t i {0}; i < m_dim; ++i) {
        double dist = compute_distance(new_point, m_points[i].first);
        if (dist < m_min_dist) {
            m_min_dist = dist;
        }
    }
}

NelderMead::NelderMead(NelderMeadParams params,
                       const std::vector<std::vector<double>>& initial_simplex,
                       const std::vector<double>& min_bounds,
                       const std::vector<double>& max_bounds,
                       unsigned int max_iter,
                       double dist_tol,
                       double std_tol)
    : Optimizer(min_bounds, max_bounds), m_stage(NelderMead::Stage::NEW_SIMPLEX), m_params(params),
      m_dim(initial_simplex.size() - 1), m_current_simplex(m_dim), m_max_iter(max_iter),
      m_dist_tol(dist_tol), m_std_tol(std_tol), m_last_reflect(), m_new_simplex_index(0),
      m_new_simplex(initial_simplex.begin(), initial_simplex.end())
{
    if (m_min_bounds.size() != m_dim) {
        throw std::runtime_error("Minimum bounds must be the same size as number of dimensions.");
    }
    if (m_max_bounds.size() != m_dim) {
        throw std::runtime_error("Minimum bounds must be the same size as number of dimensions.");
    }
}

std::vector<double> NelderMead::reflect()
{
    m_stage = NelderMead::Stage::REFLECT;
    auto centroid = m_current_simplex.compute_centroid();
    auto& worst_point = m_current_simplex.get_point(m_dim);
    auto reflected_point = std::vector<double>(centroid.size());
    for (size_t i {0}; i < m_dim; ++i) {
        reflected_point[i] = centroid[i] + m_params.alpha * (centroid[i] - worst_point[i]);
    }
    return reflected_point;
}

std::vector<double> NelderMead::expand()
{
    m_stage = NelderMead::Stage::EXPAND;
    auto centroid = m_current_simplex.compute_centroid();
    m_last_reflect = std::make_pair(m_point, m_objective);
    auto expanded_point = std::vector<double>(centroid.size());
    for (size_t i {0}; i < m_dim; ++i) {
        expanded_point[i] = centroid[i] + m_params.gamma * (m_point[i] - centroid[i]);
    }
    return expanded_point;
}

std::vector<double> NelderMead::outside_contract()
{
    m_stage = NelderMead::Stage::OUTSIDE_CONTRACT;
    m_last_reflect = std::make_pair(m_point, m_objective);
    auto centroid = m_current_simplex.compute_centroid();
    auto contracted_point = std::vector<double>(centroid.size());
    for (size_t i {0}; i < m_dim; ++i) {
        contracted_point[i] = centroid[i] + m_params.rho * (m_point[i] - centroid[i]);
    }
    return contracted_point;
}

std::vector<double> NelderMead::inside_contract()
{
    m_stage = NelderMead::Stage::INSIDE_CONTRACT;
    auto centroid = m_current_simplex.compute_centroid();
    auto& worst_point = m_current_simplex.get_point(m_dim);
    auto contracted_point = std::vector<double>(centroid.size());
    for (size_t i {0}; i < m_dim; ++i) {
        contracted_point[i] = centroid[i] + m_params.rho * (worst_point[i] - centroid[i]);
    }
    return contracted_point;
}

std::vector<double> NelderMead::shrink()
{
    m_stage = NelderMead::Stage::NEW_SIMPLEX;
    const auto best_point = m_current_simplex[0];
    m_new_simplex.clear();
    m_new_simplex.push_back(best_point.first);
    for (size_t i {1}; i < m_current_simplex.size(); ++i) {
        auto& old_point = m_current_simplex.get_point(i);
        auto new_point = std::vector<double>();
        for (size_t j {0}; j < m_dim; ++j) {
            new_point.push_back(old_point[j]
                                + m_params.sigma * (old_point[j] - best_point.first[j]));
        }
        m_new_simplex.push_back(new_point);
    }
    m_current_simplex = OrderedSimplex(m_dim);
    m_current_simplex.add(best_point.first, best_point.second);
    m_new_simplex_index = 2;
    return m_new_simplex[1];
}

void NelderMead::internal_next_point()
{
    switch (m_stage) {
    case NelderMead::Stage::NEW_SIMPLEX:
        if (m_new_simplex_index != 0) {
            m_current_simplex.add(m_point, m_objective);
        }
        if (m_new_simplex_index == m_new_simplex.size()) {
            m_point = reflect();
        } else {
            m_point = m_new_simplex[m_new_simplex_index];
            ++m_new_simplex_index;
        }
        break;
    case NelderMead::Stage::REFLECT:
        if (m_objective < m_current_simplex.get_objective(0)) {
            m_point = expand();
        } else if (m_objective < m_current_simplex.get_objective(m_dim - 1)) {
            m_current_simplex.add(m_point, m_objective);
            m_point = reflect();
        } else if (m_objective < m_current_simplex.get_objective(m_dim)) {
            m_point = outside_contract();
        } else {
            m_point = inside_contract();
        }
        break;
    case NelderMead::Stage::EXPAND:
        if (m_objective < m_last_reflect.second) {
            m_current_simplex.add(m_point, m_objective);
        } else {
            m_current_simplex.add(m_last_reflect.first, m_last_reflect.second);
        }
        m_point = reflect();
        break;
    case NelderMead::Stage::OUTSIDE_CONTRACT:
        if (m_objective < m_last_reflect.second) {
            m_current_simplex.add(m_point, m_objective);
            m_point = reflect();
        } else {
            m_point = shrink();
        }
        break;
    case NelderMead::Stage::INSIDE_CONTRACT:
        if (m_objective < m_current_simplex.get_objective(m_dim)) {
            m_current_simplex.add(m_point, m_objective);
            m_point = reflect();
        } else {
            m_point = shrink();
        }
        break;
    default:
        break;
    }
}

bool NelderMead::terminate() const
{
    if (m_stage == NelderMead::Stage::NEW_SIMPLEX) {
        return false;
    }
    return m_count > m_max_iter || m_current_simplex.get_objective_std() < m_std_tol
           || m_current_simplex.get_min_dist() < m_dist_tol;
}

std::unique_ptr<Optimizer> NelderMead::clone() const
{
    return std::make_unique<NelderMead>(*this);
}
}} // end namespace pgop::optimize
