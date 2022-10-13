#include <functional>
#include <iterator>

#include <pybind11/stl.h>

#include "MonteCarlo.h"

namespace pgop { namespace optimize {
TrialMoveGenerator::TrialMoveGenerator(long unsigned int seed,
                                       double diameter,
                                       double kT,
                                       unsigned int dim)
    : m_dim {dim}, m_diameter {diameter}, m_rng_engine {seed},
      m_uniform_dist {-diameter, diameter}, m_inv_kT {1 / kT}, m_acceptance_dist {0, 1}
{
}

std::vector<double> TrialMoveGenerator::getSample()
{
    auto random_vector = std::vector<double>();
    random_vector.reserve(m_dim);
    getSample(random_vector);
    return random_vector;
}

void TrialMoveGenerator::getSample(std::vector<double>& buf)
{
    for (size_t i {0}; i < m_dim; ++i) {
        buf.emplace_back(this->m_uniform_dist(m_rng_engine));
    }
}

std::vector<double> TrialMoveGenerator::generate()
{
    auto random_vector = std::vector<double>();
    random_vector.reserve(m_dim);
    generate(random_vector);
    return random_vector;
}

void TrialMoveGenerator::generate(std::vector<double>& buf)
{
    double norm = 0;
    do {
        buf.clear();
        getSample(buf);
        norm = std::transform_reduce(buf.cbegin(),
                                     buf.cend(),
                                     buf.cbegin(),
                                     0,
                                     std::plus {},
                                     std::multiplies {});
    } while (norm > 1);
}

bool TrialMoveGenerator::accept(double neg_energy_change)
{
    bool acc = m_acceptance_dist(m_rng_engine) < std::exp(neg_energy_change * m_inv_kT);
    return acc;
}

MonteCarlo::MonteCarlo(const std::vector<double>& min_bounds,
                       const std::vector<double>& max_bounds,
                       const std::pair<std::vector<double>, double>& initial_point,
                       double kT,
                       double max_move_size,
                       long unsigned int seed,
                       unsigned int max_iter)
    : Optimizer(min_bounds, max_bounds), m_best_point(initial_point),
      m_current_point(initial_point), m_trial_point(initial_point.first.size()),
      m_move_generator(seed, max_move_size, kT, initial_point.first.size()),
      m_move_buf(initial_point.first.size()), m_max_iter {max_iter}, m_cnt {0}
{
}

void MonteCarlo::record_objective(double objective)
{
    if (objective > m_current_point.second
        || !m_move_generator.accept(m_current_point.second - objective)) {
        return;
    }
    m_current_point.first = m_trial_point;
    m_current_point.second = objective;
    if (objective < m_best_point.second) {
        m_best_point.first = m_trial_point;
        m_best_point.second = objective;
    }
}

std::vector<double> MonteCarlo::next_point()
{
    m_cnt += 1;
    m_move_generator.generate(m_move_buf);
    m_trial_point.clear();
    std::transform(m_current_point.first.cbegin(),
                   m_current_point.first.cend(),
                   m_move_buf.cbegin(),
                   std::back_inserter(m_trial_point),
                   std::plus {});
    clip_point(m_trial_point);
    return m_trial_point;
}

bool MonteCarlo::terminate() const
{
    return m_cnt > m_max_iter;
}

std::pair<std::vector<double>, double> MonteCarlo::get_optimum() const
{
    return m_best_point;
}
std::unique_ptr<Optimizer> MonteCarlo::clone() const
{
    return std::make_unique<MonteCarlo>(*this);
}

void export_monte_carlo(py::module& m)
{
    py::class_<MonteCarlo, Optimizer, std::shared_ptr<MonteCarlo>>(m, "MonteCarlo")
        .def(py::init<const std::vector<double>&,
                      const std::vector<double>&,
                      const std::pair<std::vector<double>, double>&,
                      double,
                      double,
                      long unsigned int,
                      unsigned int>());
}
}} // namespace pgop::optimize
