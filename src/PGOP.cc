// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <cmath>
#include <iterator>
#include <string>

#include "PGOP.h"
#include "locality.h"
#include "util/Metrics.h"
#include "util/Threads.h"

namespace spatula {

PGOP::PGOP(const std::vector<const double*> R_ij_data,
           const size_t n_symmetries,
           std::shared_ptr<optimize::Optimizer>& optimizer,
           std::vector<size_t> group_sizes,
           unsigned int mode,
           bool compute_per_operator)
    : m_n_symmetries(n_symmetries), m_Rij(R_ij_data), m_group_sizes(group_sizes),
      m_optimize(optimizer), m_mode(mode), m_compute_per_operator(compute_per_operator)
{
    // size_t current_data_offset = 0;
    // for (size_t i = 0; i < m_n_symmetries; ++i) {
    //     std::vector<double> vec;
    //     vec.reserve(R_ij_size);
    //     for (size_t j = 0; j < R_ij_size; ++j) {
    //         vec.push_back(R_ij_data[current_data_offset + j]);
    //     }
    //     m_Rij.emplace_back(std::move(vec));
    //     current_data_offset += R_ij_size;
    // }
}

std::tuple<std::vector<double>, std::vector<data::Quaternion>>
PGOP::compute(const double* distances,
              const double* weights,
              const int* num_neighbors,
              const double* sigmas,
              const size_t N_particles_in_neighbors) const
{
    const auto neighborhoods
        = Neighborhoods(N_particles_in_neighbors, num_neighbors, weights, distances, sigmas);
    const size_t N_particles = N_particles_in_neighbors;
    size_t ops_per_particle = m_n_symmetries;
    if (m_compute_per_operator) {
        for (const size_t group_size : m_group_sizes) {
            ops_per_particle += group_size / 9;
        }
        // for (const auto& R_ij : m_Rij) {
        //     ops_per_particle += R_ij.size() / 9;
        // }
    }

    std::vector<double> op_values(N_particles * ops_per_particle);
    std::vector<data::Quaternion> rotation_values(N_particles * ops_per_particle);

    const auto loop_func = [&](const size_t start_idx, const size_t stop_idx) {
        for (size_t i = start_idx; i < stop_idx; ++i) {
            const size_t current_particle_offset = i * ops_per_particle;
            if (neighborhoods.getNeighborCount(i) == 0) {
                for (size_t j {0}; j < ops_per_particle; ++j) {
                    op_values[current_particle_offset + j]
                        = std::numeric_limits<double>::quiet_NaN();
                    rotation_values[current_particle_offset + j] = data::Quaternion(1, 0, 0, 0);
                }
                continue;
            }
            auto neighborhood = neighborhoods.getNeighborhood(i);
            const auto particle_op_rot = this->compute_particle(neighborhood);

            const auto& particle_ops = std::get<0>(particle_op_rot);
            const auto& particle_rots = std::get<1>(particle_op_rot);

            for (size_t j = 0; j < ops_per_particle; ++j) {
                op_values[current_particle_offset + j] = particle_ops[j];
                rotation_values[current_particle_offset + j] = particle_rots[j];
            }
        }
    };
    execute_func(loop_func, N_particles);

    return std::make_tuple(op_values, rotation_values);
}

std::tuple<std::vector<double>, std::vector<data::Quaternion>>
PGOP::compute_particle(LocalNeighborhood& neighborhood_original) const
{
    // Optimized PGOP value for each group
    auto spatula = std::vector<double>();
    spatula.reserve(m_Rij.size());

    // Store the single, optimal quaternion for each group
    auto rotations = std::vector<data::Quaternion>();
    rotations.reserve(m_Rij.size());
    // Loop over the point groups
    // for (const auto& R_ij : m_Rij) {
    for (size_t group_idx = 0; group_idx < m_Rij.size(); ++group_idx) {
        auto R_ij = m_Rij[group_idx];
        // make a copy of the neighborhood to avoid modifying the original
        auto neighborhood = neighborhood_original;
        const auto result = compute_symmetry(neighborhood, R_ij);
        spatula.emplace_back(std::get<0>(result));
        const auto quat = data::Quaternion(std::get<1>(result));
        rotations.emplace_back(quat);
        if (m_compute_per_operator) {
            auto neighborhood = neighborhood_original;
            neighborhood.rotate(std::get<1>(result));
            // loop over every operator; each operator is a 3x3 matrix so size 9
            for (size_t i = 0; i < m_group_sizes[i]; i += 9) {
                // Compute the PGOP value for a single operator in our group
                const auto particle_operator_op
                    = compute_pgop(neighborhood, std::span<const double, 9>(R_ij + i, 9));
                spatula.emplace_back(particle_operator_op);
                rotations.emplace_back(quat);
            }
        }
    }
    return std::make_tuple(std::move(spatula), std::move(rotations));
}

std::tuple<double, data::Vec3> PGOP::compute_symmetry(LocalNeighborhood& neighborhood,
                                                      const double* R_ij) const
{
    auto opt = m_optimize->clone();
    while (!opt->terminate()) {
        neighborhood.rotate(opt->next_point());
        const auto particle_op = compute_pgop(neighborhood, std::span<const double, 9>(R_ij, 9));
        opt->record_objective(-particle_op);
    }
    const auto optimum = opt->get_optimum();
    // op value is negated to get the correct value, because optimization scheme is
    // minimization not maximization!
    return std::make_tuple(-optimum.second, optimum.first);
}

double PGOP::compute_pgop(LocalNeighborhood& neighborhood,
                          const std::span<const double, 9> R_ij) const
{
    const auto positions = neighborhood.rotated_positions;
    const auto sigmas = neighborhood.sigmas;
    double overlap = 0.0;
    // loop over the R_ij. Each 3x3 segment is a symmetry operation
    // matrix. Each matrix should be applied to each point in positions.
    for (size_t i {0}; i < R_ij.size(); i += 9) {
        // loop over positions
        for (size_t j {0}; j < positions.size(); ++j) {
            // symmetrized position is obtained by multiplying the operator with the position
            auto symmetrized_position = data::Vec3(0, 0, 0);
            // create 3x3 double loop for matrix vector multiplication
            for (size_t k {0}; k < 3; ++k) {
                for (size_t l {0}; l < 3; ++l) {
                    symmetrized_position[k] += R_ij[i + k * 3 + l] * positions[j][l];
                }
            }
            // compute overlap with every point in the positions
            double max_res = 0.0;
            for (size_t m {0}; m < positions.size(); ++m) {
                double BC = 0;
                if (m_mode == 0) {
                    BC = util::compute_Bhattacharyya_coefficient_gaussian(positions[m],
                                                                          symmetrized_position,
                                                                          sigmas[j],
                                                                          sigmas[m]);
                } else {
                    BC = util::compute_Bhattacharyya_coefficient_fisher(positions[m],
                                                                        symmetrized_position,
                                                                        sigmas[j],
                                                                        sigmas[m]);
                }
                if (BC > max_res)
                    max_res = BC;
            }
            overlap += max_res;
        }
    }
    // cast to double to avoid integer division
    const auto normalization = static_cast<double>(positions.size() * R_ij.size()) / 9.0;
    return overlap / normalization;
}

void PGOP::execute_func(std::function<void(size_t, size_t)> func, size_t N) const
{
    // Enable py-spy profiling through serial mode.
    if (util::ThreadPool::get().get_num_threads() == 1) {
        util::ThreadPool::get().serial_compute<void, size_t>(0, N, func);
    } else {
        auto& pool = util::ThreadPool::get().get_pool();
        pool.push_loop(0, N, func, 2 * pool.get_thread_count());
        pool.wait_for_tasks();
    }
}

} // End namespace spatula
