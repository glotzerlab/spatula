#pragma once

#include <memory>
#include <random>

#include "Optimize.h"

namespace pgop { namespace optimize {

class RandomSearch : public Optimizer {
    public:
    RandomSearch(unsigned int max_iter, long unsigned int seed);
    ~RandomSearch() override = default;
    /// Returns whether or not convergence or termination conditions have been met.
    bool terminate() const override;
    /// Create a clone of this optimizer
    std::unique_ptr<Optimizer> clone() const override;

    /// Set the next point to compute the objective for to m_point.
    void internal_next_point() override;

    long unsigned int getSeed() const;

    void setSeed(long unsigned int seed);

    unsigned int getMaxIter() const;

    void setMaxIter(unsigned int iter);

    private:
    unsigned int m_max_iter;
    long unsigned int m_seed;
    std::mt19937_64 m_rng;
    std::normal_distribution<double> m_normal_dist;
};

void export_random_search(py::module& m);
}} // namespace pgop::optimize
