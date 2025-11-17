// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once
#include <memory>
#include <tuple>
#include <vector>

#include "data/Quaternion.h"
#include "optimize/Optimize.h"
#include "util/Metrics.h"
#include "util/Util.h"

namespace spatula {

/**
 * @brief storage for neighbor positions, weights, and rotated positions.
 *
 * This struct is helpful is organizing operations on the BOD's neighbors when optimizing over
 * SO(3), by keeping all the data together.
 */
struct LocalNeighborhood {
    LocalNeighborhood(std::vector<data::Vec3>&& positions_,
                      std::vector<double>&& weights_,
                      std::vector<double>&& sigmas_);

    void rotate(const data::Vec3& q);

    /// neighbor bonds
    const std::vector<data::Vec3> positions;
    /// neighbor weights
    const std::vector<double> weights;
    const std::vector<double> sigmas;
    /// Storage for the current positions under a given rotation used in optimization.
    std::vector<data::Vec3> rotated_positions;
};

/**
 * @brief Small helper class for storing and accessing neighbor list data from Python.
 *
 * Warning: This class consumes the raw arrays from Python and requires that these arrays outlive
 * class instances.
 */
class Neighborhoods {
    public:
    Neighborhoods(size_t N,
                  const int* neighbor_counts,
                  const double* weights,
                  const double* distance,
                  const double* sigmas);

    /// Get the neighbors for point i.
    LocalNeighborhood getNeighborhood(size_t i) const;
    /// Get the neighbor weight for point i.
    std::vector<double> getWeights(size_t i) const;
    /// Get the sigmas for each neighbor bond
    std::vector<double> getSigmas(size_t i) const;
    /// Get the number of neighbors for point i.
    int getNeighborCount(size_t i) const;

    private:
    /// Number of points with neighbors
    const size_t m_N;
    /// The number of neighbors for each point
    const int* m_neighbor_counts;
    /// The distance vector of each neighbor
    const double* m_distances;
    /// The weights for each neighbor bond
    const double* m_weights;
    /// The sigmas for each neighbor bond
    const double* m_sigmas;
    /// The offsets to index into the distances array
    std::vector<size_t> m_neighbor_offsets;
};

/**
 * @brief Store for the optimal PGOP values and rotations
 *
 * This simplifies the setting of the results from the optimization, and the use of numpy arrays in
 * the code.
 */
struct PGOPStore {
    PGOPStore(size_t N_particles, size_t N_symmetries);
    /// Number of point group symmetries to compute
    size_t N_syms;
    /// The optimized value of PGOP for each point group
    std::vector<double> op;
    /// The optimal rotations used to obtain the maximum PGOP as quaternions.
    std::vector<double> rotations;

    /// Add a single point's set of PGOP and rotation values
    void addOp(size_t i, const std::tuple<std::vector<double>, std::vector<data::Quaternion>>& op_);
    /// Store 0's for point i. This is used when no neighbors for a point exist.
    void addNull(size_t i);
    /// Return a tuple of the two arrays op and rotations.
    std::pair<std::vector<double>, std::vector<double>> getArrays();
};

/**
 * @brief Central class, computes PGOP for provided points.
 *
 * Compute uses many levels of functions to compute PGOP these should be inlined for performance.
 * The nestedness is to make each function comprehendible by itself and not too long.
 */
class PGOP {
    public:
    PGOP(const std::vector<std::vector<double>>& R_ij,
         std::shared_ptr<optimize::Optimizer>& optimizer,
         const unsigned int mode,
         bool compute_per_operator);

    /**
     * @brief Root function for computing PGOP for a set of points.
     *
     * @param distances An array of distance vectors for neighbors
     * @param weights An array of neighbor weights. For unweighted PGOP use an array of 1s.
     * @param num_neighboors An array of the number of neighbor for each point.
     *
     */
    std::pair<std::vector<double>, std::vector<double>> compute(size_t N_points,
                                                                 const double* distances,
                                                                 const double* weights,
                                                                 const int* num_neighbors,
                                                                 const double* sigmas) const;

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
    std::tuple<double, data::Vec3> compute_symmetry(LocalNeighborhood& neighborhood,
                                                    const std::vector<double>& R_ij) const;

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
    double compute_pgop(LocalNeighborhood& neighborhood, const std::vector<double>& R_ij) const;

    /**
     * Helper function to better handle both single threaded and multithreaded behavior. In single
     * threaded behavior, we need to not use the thread_pool library to get readable profiles from
     * profilerslike py-spy.
     */
    void execute_func(std::function<void(size_t, size_t)> func, size_t N) const;

    /// The number of symmetries that PGOP is being computed for.
    unsigned int m_n_symmetries;
    /// The Wigner D matrices for each point group symmetry
    std::vector<std::vector<double>> m_Rij;
    /// Optimizer to find the optimal rotation for each point and symmetry.
    std::shared_ptr<const optimize::Optimizer> m_optimize;
    /// The mode of the PGOP computation.
    unsigned int m_mode;
    // Whether to compute the PGOP for each operator.
    bool m_compute_per_operator;
};

void export_spatula_class(py::module& m, const std::string& name);

void export_spatula(py::module& m);
} // End namespace spatula
