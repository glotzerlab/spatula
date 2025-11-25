// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <complex>
#include <memory>
#include <tuple>
#include <vector>

#include "data/Quaternion.h"
#include "locality.h" // Include locality.h for LocalNeighborhoodBOOBOO
#include "optimize/Optimize.h"
#include "util/Metrics.h"
#include "util/QlmEval.h"
#include "util/Util.h"

namespace spatula {

/**
 * @brief Central class, computes BOOSOP for provided points.
 *
 * Compute uses many levels of functions to compute BOOSOP these should be inlined for performance.
 * The nestedness is to make each function comprehendible by itself and not too long.
 */
template<typename distribution_type> class BOOSOP {
    public:
    BOOSOP(const std::vector<std::vector<std::complex<double>>>& D_ij,
           std::shared_ptr<optimize::Optimizer>& optimizer,
           typename distribution_type::param_type distribution_params);

    /**
     * @brief Root function for computing BOOSOP for a set of points.
     *
     * @param distances An array of distance vectors for neighbors
     * @param weights An array of neighbor weights. For unweighted BOOSOP use an array of 1s.
     * @param num_neighboors An array of the number of neighbor for each point.
     * @param m The degree of Gauss-Legendre quadrature to use when computing Qlms. This is not used
     * directly expect for normalizing the quadrature values.
     * @param ylms 2D array of spherical harmonic values for all points in the Gauss-Legendre
     * quadrature as well as for every combination of m (spherical harmonic number) and l upto a
     * maximum l. The first dimension is the harmonic numbers and the second is the quadrature
     * points.
     * @param quad_positions The positions of the Gauss-Legendre quadrature.
     * @param quad_weights The weights associated with the Gauss-Legendre quadrature points.
     *
     */
    std::tuple<std::vector<double>, std::vector<data::Quaternion>>
    compute(const double* distances,
            const double* weights,
            const int* num_neighbors,
            size_t N_particles,
            const unsigned int m,
            const std::complex<double>* ylms,
            size_t ylms_shape_0,
            const double* quad_positions,
            size_t quad_positions_shape_0,
            const double* quad_weights) const;

    /**
     * @brief Compute BOOSOP at given rotations for each point.
     *
     * This method is primarily for computing BOOSOP after an initial optimization was performed and
     * a calculation at higher quadrature and spherical harmonic number is desired.
     *
     * @param distances An array of distance vectors for neighbors
     * @param distances An array of quaternion rotations to use for computing BOOSOP.
     * @param weights An array of neighbor weights. For unweighted BOOSOP use an array of 1s.
     * @param num_neighboors An array of the number of neighbor for each point.
     * @param m The degree of Gauss-Legendre quadrature to use when computing Qlms. This is not used
     * directly expect for normalizing the quadrature values.
     * @param ylms 2D array of spherical harmonic values for all points in the Gauss-Legendre
     * quadrature as well as for every combination of m (spherical harmonic number) and l upto a
     * maximum l. The first dimension is the harmonic numbers and the second is the quadrature
     * points.
     * @param quad_positions The positions of the Gauss-Legendre quadrature.
     * @param quad_weights The weights associated with the Gauss-Legendre quadrature points.
     *
     */
    std::vector<double> refine(const double* distances,
                               const double* rotations,
                               const double* weights,
                               const int* num_neighbors,
                               size_t N_particles,
                               const unsigned int m,
                               const std::complex<double>* ylms,
                               size_t ylms_shape_0,
                               const double* quad_positions,
                               size_t quad_positions_shape_0,
                               const double* quad_weights) const;

    private:
    /**
     * @brief Compute the optimal BOOSOP and rotation for all points groups for a given point.
     *
     *
     * @param neighborhood the local neighborhood (weights, positions) to compute BOOSOP for
     * @param qlm_eval The object to evaluate the spherical harmonic expansion for the BOD of
     * neighborhood
     * @param qlm_buf The buffer for the symmetrized and unsymmetrized BOD spherical harmonic
     * expansions
     *
     * @returns the optimized BOOSOP value and the optimal rotation for the given point for all
     * specified point group symmetries.
     */
    std::tuple<std::vector<double>, std::vector<data::Quaternion>>
    compute_particle(LocalNeighborhoodBOOBOO& neighborhood,
                     const util::QlmEval& qlm_eval,
                     util::QlmBuf& qlm_buf) const;

    /**
     * @brief Compute the optimal BOOSOP and rotation for a given point group symmetry.
     *
     *
     * @param neighborhood the local neighborhood (weights, positions) to compute BOOSOP for
     * @param D_ij The Wigner D matrix for the given point group
     * @param qlm_eval The object to evaluate the spherical harmonic expansion for the BOD of
     * neighborhood
     * @param qlm_buf The buffer for the symmetrized and unsymmetrized BOD spherical harmonic
     * expansions
     *
     * @returns the optimized BOOSOP value and the optimal rotation for the given symmetry.
     */
    std::tuple<double, data::Quaternion>
    compute_symmetry(LocalNeighborhoodBOOBOO& neighborhood,
                     const std::vector<std::complex<double>>& D_ij,
                     const util::QlmEval& qlm_eval,
                     util::QlmBuf& qlm_buf) const;

    /**
     * @brief Compute the BOOSOP for a set point group symmetry and rotation.
     *
     * This is the most barebones of the algorithm. No optimization is done here just a direct
     * calculation.
     *
     * @param neighborhood the local neighborhood (weights, rotated positions) to compute BOOSOP for
     * @param D_ij The Wigner D matrix for the given point group
     * @param qlm_eval The object to evaluate the spherical harmonic expansion for the BOD of
     * neighborhood
     * @param qlm_buf The buffer for the symmetrized and unsymmetrized BOD spherical harmonic
     * expansions
     *
     * @returns The BOOSOP value.
     */
    double compute_BOOSOP(LocalNeighborhoodBOOBOO& neighborhood,
                          const std::vector<std::complex<double>>& D_ij,
                          const util::QlmEval& qlm_eval,
                          util::QlmBuf& qlm_buf) const;

    /**
     * Helper function to better handle both single threaded and multithreaded behavior. In single
     * threaded behavior, we need to not use the thread_pool library to get readable profiles from
     * profilerslike py-spy.
     */
    void execute_func(std::function<void(size_t, size_t)> func, size_t N) const;

    /// The type of distribution to use for the BOD.
    distribution_type m_distribution;
    /// The number of symmetries that BOOSOP is being computed for.
    unsigned int m_n_symmetries;
    /// The Wigner D matrices for each point group symmetry
    std::vector<std::vector<std::complex<double>>> m_Dij;
    /// Optimizer to find the optimal rotation for each point and symmetry.
    std::shared_ptr<const optimize::Optimizer> m_optimize;
};

} // End namespace spatula
