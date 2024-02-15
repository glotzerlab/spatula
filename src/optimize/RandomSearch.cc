#include "RandomSearch.h"
#include "../data/Quaternion.h"

namespace pgop { namespace optimize {

RandomSearch::RandomSearch(unsigned int max_iter, long unsigned int seed)
    : Optimizer(), m_iterations(max_iter), m_seed(seed), m_rng(seed), m_normal_dist(0, 1.0)
{
}

bool RandomSearch::terminate() const
{
    return m_count > m_iterations;
}

std::unique_ptr<Optimizer> RandomSearch::clone() const
{
    return std::make_unique<RandomSearch>(*this);
}

void RandomSearch::internal_next_point()
{
    data::Quaternion q(m_normal_dist(m_rng),
                       m_normal_dist(m_rng),
                       m_normal_dist(m_rng),
                       m_normal_dist(m_rng));
    q.normalize();
    m_point = q.to_axis_angle_3D();
}

long unsigned int RandomSearch::getSeed() const
{
    return m_seed;
}

void RandomSearch::setSeed(long unsigned int seed)
{
    m_seed = seed;
    m_rng.seed(seed);
}

unsigned int RandomSearch::getIterations() const
{
    return m_iterations;
}

void RandomSearch::setIterations(unsigned int iter)
{
    m_iterations = iter;
}

void export_random_search(py::module& m)
{
    py::class_<RandomSearch, Optimizer, std::shared_ptr<RandomSearch>>(m, "RandomSearch")
        .def(py::init<unsigned int, unsigned int>())
        .def_property("max_iter", &RandomSearch::getIterations, &RandomSearch::setIterations)
        .def_property("seed", &RandomSearch::getSeed, &RandomSearch::setSeed);
}
}} // namespace pgop::optimize
