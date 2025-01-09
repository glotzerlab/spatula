// Copyright (c) 2010-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once
#include <memory>
#include <tuple>
#include <vector>

#include <complex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "data/Quaternion.h"
#include "optimize/Optimize.h"
#include "util/Metrics.h"
#include "util/QlmEval.h"
#include "util/Util.h"

namespace py = pybind11;

namespace spatula {

/**
 * @brief storage for neighbor positions, weights, and rotated positions.
 *
 * This struct is helpful is organizing operations on the BOD's neighbors when optimizing over
 * SO(3), by keeping all the data together.
 */
struct LocalNeighborhoodBOOBOO {
    LocalNeighborhoodBOOBOO(std::vector<data::Vec3>&& positions_, std::vector<double>&& weights_);

    void rotate(const data::Vec3& q);

    /// BOD neighbor bonds
    const std::vector<data::Vec3> positions;
    /// BOD neighbor weights
    const std::vector<double> weights;
    /// Storage for the current positions under a given rotation used in optimization.
    std::vector<data::Vec3> rotated_positions;
};

/**
 * @brief Small helper class for storing and accessing neighbor list data from Python.
 *
 * Warning: This class consumes the raw arrays from Python and requires that these arrays outlive
 * class instances.
 */
class NeighborhoodBOOs {
    public:
    NeighborhoodBOOs(size_t N,
                     const int* neighbor_counts,
                     const double* weights,
                     const double* distance);

    /// Get the neighbors for point i.
    LocalNeighborhoodBOOBOO getNeighborhoodBOO(size_t i) const;
    /// Get the normalized neighbor distance vectors for point i.
    std::vector<data::Vec3> getNormalizedDistances(size_t i) const;
    /// Get the neighbor weight for point i.
    std::vector<double> getWeights(size_t i) const;
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
    /// The offsets to index into the distances array
    std::vector<size_t> m_neighbor_offsets;
};

/**
 * @brief Store for the optimal BOOSOP values and rotations
 *
 * This simplifies the setting of the results from the optimization, and the use of numpy arrays in
 * the code.
 */
struct BOOSOPStore {
    BOOSOPStore(size_t N_particles, size_t N_symmetries);
    /// Number of point group symmetries to compute
    size_t N_syms;
    /// The optimized value of BOOSOP for each point group
    py::array_t<double> op;
    /// The optimal rotations used to obtain the maximum BOOSOP as quaternions.
    py::array_t<double> rotations;

    /// Add a single point's set of BOOSOP and rotation values
    void addOp(size_t i, const std::tuple<std::vector<double>, std::vector<data::Quaternion>>& op_);
    /// Store 0's for point i. This is used when no neighbors for a point exist.
    void addNull(size_t i);
    /// Return a tuple of the two arrays op and rotations.
    py::tuple getArrays();

    private:
    /// Fast access to op
    py::detail::unchecked_mutable_reference<double, 2> u_op;
    /// Fast access to rotations
    py::detail::unchecked_mutable_reference<double, 3> u_rotations;
};

/**
 * @brief Central class, computes BOOSOP for provided points.
 *
 * Compute uses many levels of functions to compute BOOSOP these should be inlined for performance.
 * The nestedness is to make each function comprehendible by itself and not too long.
 */
template<typename distribution_type> class BOOSOP {
    public:
    BOOSOP(const py::array_t<std::complex<double>> D_ij,
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
    py::tuple compute(const py::array_t<double> distances,
                      const py::array_t<double> weights,
                      const py::array_t<int> num_neighbors,
                      const unsigned int m,
                      const py::array_t<std::complex<double>> ylms,
                      const py::array_t<double> quad_positions,
                      const py::array_t<double> quad_weights) const;

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
    py::array_t<double> refine(const py::array_t<double> distances,
                               const py::array_t<double> rotations,
                               const py::array_t<double> weights,
                               const py::array_t<int> num_neighbors,
                               const unsigned int m,
                               const py::array_t<std::complex<double>> ylms,
                               const py::array_t<double> quad_positions,
                               const py::array_t<double> quad_weights) const;

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

template<typename distribution_type>
void export_BOOSOP_class(py::module& m, const std::string& name);

void export_BOOSOP(py::module& m);
} // End namespace spatula
