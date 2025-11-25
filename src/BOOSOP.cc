// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <cmath>
#include <iterator>
#include <string>

#include "BOOSOP.h"
#include "BondOrder.h"
#include "locality.h"
#include "util/Threads.h"

namespace spatula {

// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <cmath>
#include <iterator>
#include <string>

#include "BOOSOP.h"
#include "BondOrder.h"
#include "locality.h"
#include "util/Threads.h"

namespace spatula {

template<typename distribution_type>
BOOSOP<distribution_type>::BOOSOP(
    const std::vector<std::vector<std::complex<double>>>& D_ij,
    std::shared_ptr<optimize::Optimizer>& optimizer,
    typename distribution_type::param_type distribution_params)
    : m_distribution(distribution_params), m_n_symmetries(D_ij.size()), m_Dij(D_ij),
      m_optimize(optimizer)
{}

template<typename distribution_type>
std::tuple<std::vector<double>, std::vector<data::Quaternion>>
BOOSOP<distribution_type>::compute(const double* distances,
                                   const double* weights,
                                   const int* num_neighbors,
                                   size_t N_particles,
                                   const unsigned int m,
                                   const std::complex<double>* ylms,
                                   size_t ylms_shape_0,
                                   const double* quad_positions,
                                   size_t quad_positions_shape_0,
                                   const double* quad_weights) const
{
    const auto qlm_eval = util::QlmEval(
        m, quad_positions, quad_weights, ylms, quad_positions_shape_0, ylms_shape_0);
    const auto neighborhoods = NeighborhoodBOOs(N_particles, num_neighbors, weights, distances);

    std::vector<double> op_values(N_particles * m_n_symmetries);
    std::vector<data::Quaternion> rotation_values(N_particles * m_n_symmetries);

    const auto loop_func = [&](const size_t start, const size_t stop) {
        auto qlm_buf = util::QlmBuf(qlm_eval.getNlm());
        for (size_t i = start; i < stop; ++i)
        {
            const size_t current_particle_offset = i * m_n_symmetries;
            if (neighborhoods.getNeighborCount(i) == 0)
            {
                for (size_t j {0}; j < m_n_symmetries; ++j)
                {
                    op_values[current_particle_offset + j] = 0;
                    rotation_values[current_particle_offset + j] = data::Quaternion(1, 0, 0, 0);
                }
                continue;
            }
            auto neighborhood = neighborhoods.getNeighborhoodBOO(i);
            const auto particle_op_rot
                = this->compute_particle(neighborhood, qlm_eval, qlm_buf);

            const auto& values = std::get<0>(particle_op_rot);
            const auto& rots = std::get<1>(particle_op_rot);
            for (size_t j {0}; j < m_n_symmetries; ++j)
            {
                op_values[current_particle_offset + j] = values[j];
                rotation_values[current_particle_offset + j] = rots[j];
            }
        }
    };
    execute_func(loop_func, N_particles);
    return std::make_tuple(op_values, rotation_values);
}

template<typename distribution_type>
std::vector<double> BOOSOP<distribution_type>::refine(const double* distances,
                                                      const double* rotations,
                                                      const double* weights,
                                                      const int* num_neighbors,
                                                      size_t N_particles,
                                                      const unsigned int m,
                                                      const std::complex<double>* ylms,
                                                      size_t ylms_shape_0,
                                                      const double* quad_positions,
                                                      size_t quad_positions_shape_0,
                                                      const double* quad_weights) const
{
    const auto qlm_eval = util::QlmEval(
        m, quad_positions, quad_weights, ylms, quad_positions_shape_0, ylms_shape_0);
    const auto neighborhoods = NeighborhoodBOOs(N_particles, num_neighbors, weights, distances);
    std::vector<double> op_store(N_particles * m_n_symmetries);

    const auto loop_func = [&](const size_t start, const size_t stop) {
        auto qlm_buf = util::QlmBuf(qlm_eval.getNlm());
        for (size_t i = start; i < stop; ++i)
        {
            if (neighborhoods.getNeighborCount(i) == 0)
            {
                for (size_t j {0}; j < m_n_symmetries; ++j)
                {
                    op_store[i * m_n_symmetries + j] = 0;
                }
                continue;
            }
            auto neighborhood = neighborhoods.getNeighborhoodBOO(i);
            for (size_t j {0}; j < m_n_symmetries; ++j)
            {
                const size_t rot_idx = (i * m_n_symmetries + j) * 4;
                const auto rot = data::Quaternion(rotations[rot_idx],
                                                  rotations[rot_idx + 1],
                                                  rotations[rot_idx + 2],
                                                  rotations[rot_idx + 3])
                                     .to_axis_angle_3D();
                neighborhood.rotate(rot);
                op_store[i * m_n_symmetries + j]
                    = this->compute_BOOSOP(neighborhood, m_Dij[j], qlm_eval, qlm_buf);
            }
        }
    };
    execute_func(loop_func, N_particles);
    return op_store;
}} // namespace spatula
