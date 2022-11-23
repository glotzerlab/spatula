#include <functional>
#include <iterator>

#include <pybind11/stl.h>

#include "LocalMonteCarlo.h"

namespace pgop { namespace optimize {
TrialMoveGenerator::TrialMoveGenerator(long unsigned int seed, double max_theta, double kT)
    : m_max_theta(max_theta), m_rng_engine {seed},
      m_uniform_dist {-max_theta, max_theta}, m_inv_kT {1 / kT}, m_acceptance_dist {0, 1}
{
}

data::Vec3 TrialMoveGenerator::getSample()
{
    auto random_vec3 = data::Vec3();
    // We use the acceptance rate to avoid creating another distribution. The transformation makes
    // the distribution [-1, 1].
    do {
        random_vec3.x = 2.0 * (m_acceptance_dist(m_rng_engine) - 0.5);
        random_vec3.y = 2.0 * (m_acceptance_dist(m_rng_engine) - 0.5);
        random_vec3.z = 2.0 * (m_acceptance_dist(m_rng_engine) - 0.5);
    } while (random_vec3.norm() > 1);
    random_vec3.normalize();
    return random_vec3;
}

data::Quaternion TrialMoveGenerator::generate()
{
    return data::Quaternion(getSample(), m_uniform_dist(m_rng_engine));
}

bool TrialMoveGenerator::accept(double neg_energy_change)
{
    return neg_energy_change > 0
           || m_acceptance_dist(m_rng_engine) < std::exp(neg_energy_change * m_inv_kT);
}

void TrialMoveGenerator::setSeed(long unsigned int seed)
{
    m_rng_engine.seed(seed);
}

double TrialMoveGenerator::getMaxTheta() const
{
    return m_max_theta;
}

void TrialMoveGenerator::setMaxTheta(double max_theta)
{
    m_max_theta = max_theta;
}

void TrialMoveGenerator::setkT(double kT)
{
    m_inv_kT = 1 / kT;
}

MonteCarlo::MonteCarlo(const std::pair<data::Quaternion, double>& initial_point,
                       double kT,
                       double max_theta,
                       long unsigned int seed,
                       unsigned int iterations)
    : Optimizer(), m_current_point(initial_point),
      m_move_generator(seed, max_theta, kT), m_max_iter {iterations}
{
}

void MonteCarlo::internal_next_point()
{
    if (m_move_generator.accept(m_current_point.second - m_objective)) {
        m_current_point.first = m_point;
        m_current_point.second = m_objective;
    }
    m_point = m_move_generator.generate() * m_current_point.first;
}

bool MonteCarlo::terminate() const
{
    return m_count > m_max_iter;
}

std::unique_ptr<Optimizer> MonteCarlo::clone() const
{
    return std::make_unique<MonteCarlo>(*this);
}

void MonteCarlo::specialize(unsigned int particle_index)
{
    setSeed(m_seed + particle_index);
}

double MonteCarlo::getkT() const
{
    return m_kT;
}
void MonteCarlo::setkT(double kT)
{
    m_kT = kT;
    m_move_generator.setkT(kT);
}

long unsigned int MonteCarlo::getSeed() const
{
    return m_seed;
}

void MonteCarlo::setSeed(long unsigned int seed)
{
    m_seed = seed;
    m_move_generator.setSeed(seed);
}

unsigned int MonteCarlo::getIter() const
{
    return m_max_iter;
}
void MonteCarlo::setIter(unsigned int iter)
{
    m_max_iter = iter;
}

double MonteCarlo::getMaxTheta() const
{
    return m_move_generator.getMaxTheta();
}
void MonteCarlo::setMaxTheta(double max_move_size)
{
    m_move_generator.setMaxTheta(max_move_size);
}

void export_monte_carlo(py::module& m)
{
    py::class_<MonteCarlo, Optimizer, std::shared_ptr<MonteCarlo>>(m, "QMonteCarlo")
        .def(py::init<const std::pair<data::Quaternion, double>&,
                      double,
                      double,
                      long unsigned int,
                      unsigned int>())
        .def_property("max_theta", &MonteCarlo::getMaxTheta, &MonteCarlo::setMaxTheta)
        .def_property("iterations", &MonteCarlo::getIter, &MonteCarlo::setIter)
        .def_property("kT", &MonteCarlo::getkT, &MonteCarlo::setkT)
        .def("set_seed", &MonteCarlo::setSeed);
}
}} // namespace pgop::optimize
