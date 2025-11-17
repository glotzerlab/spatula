// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include "RandomSearch.h"
#include "../data/Quaternion.h"

namespace spatula { namespace optimize {

RandomSearch::RandomSearch(unsigned int max_iter, long unsigned int seed)
    : Optimizer(), m_iterations(max_iter), m_seed(seed), m_rng(seed), m_normal_dist(0, 1.0)
{
}

bool RandomSearch::terminate() const
{
    return m_count >= m_iterations;
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

}} // namespace spatula::optimize
