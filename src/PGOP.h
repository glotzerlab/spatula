// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once
#include <memory>
#include <span>
#include <tuple>
#include <vector>
#include <functional>

#include "data/Quaternion.h"
#include "locality.h"
#include "optimize/Optimize.h"
#include "util/Metrics.h"
#include "util/Util.h"

namespace spatula {

/**
 * @brief Central class, computes PGOP for provided points.
 *
 * Compute uses many levels of functions to compute PGOP these should be inlined for performance.
 * The nestedness is to make each function comprehendible by itself and not too long.
 */
class PGOP {
    public:
    PGOP(const std::vector<const double*> R_ij_data,
         const size_t n_symmetries,
         std::shared_ptr<optimize::Optimizer>& optimizer,
         const std::vector<size_t> group_sizes,
         unsigned int mode,
         bool compute_per_operator);

    /**
     * @brief Root function for computing PGOP for a set of points.
     *
     * @param distances An array of distance vectors for neighbors
     * @param weights An array of neighbor weights. For unweighted PGOP use an array of 1s.
     * @param num_neighboors An array of the number of neighbor for each point.
     *
     */
    std::tuple<std::vector<double>, std::vector<data::Quaternion>>
    compute(const double* distances,
            const double* weights,
            const int* num_neighbors,
            const double* sigmas,
            const size_t N_particles_in_neighbors) const;

    private:
    /**
     * @brief Compute the optimal PGOP and rotation for all points groups for a given point.
     *
     *
     * @param neighborhood_original the local neighborhood (weights, positions) to compute PGOP for
     *
     * @returns the optimized PGOP value and the optimal rotation for the given point for all
     * specified point group symmetries.
     */
    std::tuple<std::vector<double>, std::vector<data::Quaternion>>
    compute_particle(LocalNeighborhood& neighborhood_original) const;

    /**
     * @brief Compute the optimal PGOP and rotation for a given point group symmetry.
     *
     *
     * @param neighborhood the local neighborhood (weights, positions) to compute PGOP for
     * @param R_ij The group action matrix for the given point group
     *
     * @returns the optimized PGOP value and the optimal rotation for the given symmetry.
     */
    std::tuple<double, data::Vec3>
    compute_symmetry(LocalNeighborhood& neighborhood, const double* R_ij, size_t group_idx) const;

    /**
     * @brief Compute the PGOP for a set point group symmetry and rotation.
     *
     * This is the most barebones of the algorithm. No optimization is done here just a direct
     * calculation.
     *
     * @param neighborhood the local neighborhood (weights, rotated positions) to compute PGOP for
     * @param R_ij The group action matrix for the given point group
     *
     * @returns The PGOP value.
     */
    double compute_pgop(LocalNeighborhood& neighborhood, const std::span<const double> R_ij) const;

    /**
     * Helper function to better handle both single threaded and multithreaded behavior. In single
     * threaded behavior, we need to not use the thread_pool library to get readable profiles from
     * profilerslike py-spy.
     */
    void execute_func(std::function<void(size_t, size_t)> func, size_t N) const;

    /// The number of symmetries that PGOP is being computed for.
    const unsigned int m_n_symmetries;
    /// The Cartesian matrices for each point group symmetry
    const std::vector<const double*> m_Rij;
    /// The number of elements in each group // TODO: divided by 9!
    const std::vector<size_t> m_group_sizes;
    /// Optimizer to find the optimal rotation for each point and symmetry.
    std::shared_ptr<const optimize::Optimizer> m_optimize;
    /// The mode of the PGOP computation.
    unsigned int m_mode;
    // Whether to compute the PGOP for each operator.
    bool m_compute_per_operator;
};

} // End namespace spatula
