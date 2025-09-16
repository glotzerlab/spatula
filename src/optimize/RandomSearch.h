// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <memory>
#include <random>

#include "Optimize.h"

namespace spatula { namespace optimize {

/**
 * @brief Perform a random search through SO(3) for optimization.
 *
 * The optimizer choses random points until m_iterations iterations. In general, Mesh should be
 * preferred since it gives deterministic results and well spaces the test points for local
 * optimization. This should not be used as the final/only optimization algorithm.
 *
 * We use a the built in Mersenne Twister for generating random numbers.
 */
class RandomSearch : public Optimizer {
    public:
    RandomSearch(unsigned int iterations, long unsigned int seed);
    ~RandomSearch() override = default;
    /// Returns whether or not convergence or termination conditions have been met.
    bool terminate() const override;
    /// Create a clone of this optimizer
    std::unique_ptr<Optimizer> clone() const override;

    /// Set the next point to compute the objective for to m_point.
    void internal_next_point() override;

    long unsigned int getSeed() const;

    void setSeed(long unsigned int seed);

    unsigned int getIterations() const;

    void setIterations(unsigned int iter);

    private:
    /// The total number of iterations to run the search
    unsigned int m_iterations;
    /// The random number seed used to generate random numbers
    long unsigned int m_seed;
    /// The Mersenne Twister object for generating random numbers
    std::mt19937_64 m_rng;
    /** A normal distribution instance for generating normally distributed data from a random number
     *  generator like std::mt19937_64.
     */
    std::normal_distribution<double> m_normal_dist;
};

void export_random_search(py::module& m);
}} // namespace spatula::optimize
