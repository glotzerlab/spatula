// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <cmath>

#include "PGOP.h"
#include "computes.h"
#include "locality.h"
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
}

std::tuple<std::vector<double>, std::vector<data::Quaternion>>
PGOP::compute(const float* distances,
              const float* weights,
              const int* num_neighbors,
              const float* sigmas,
              const size_t N_particles_in_neighbors) const
{
    const auto neighborhoods = Neighborhoods(N_particles_in_neighbors,
                                             num_neighbors,
                                             weights,
                                             distances,
                                             m_mode == 1,
                                             sigmas);
    const size_t N_particles = N_particles_in_neighbors;
    size_t ops_per_particle = m_n_symmetries;
    if (m_compute_per_operator) {
        for (const size_t group_size : m_group_sizes) {
            ops_per_particle += group_size / 9;
        }
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
        const auto result = compute_symmetry(neighborhood, R_ij, group_idx);
        spatula.emplace_back(std::get<0>(result));
        const auto quat = data::Quaternion(std::get<1>(result));
        rotations.emplace_back(quat);
        if (m_compute_per_operator) {
            auto neighborhood = neighborhood_original;
            neighborhood.rotate(std::get<1>(result));
            // loop over every operator; each operator is a 3x3 matrix so size 9
            for (size_t i = 0; i < m_group_sizes[group_idx]; i += 9) {
                // Compute the PGOP value for a single operator in our group
                const auto particle_operator_op
                    = compute_pgop(neighborhood, std::span<const double>(R_ij + i, 9));
                spatula.emplace_back(particle_operator_op);
                rotations.emplace_back(quat);
            }
        }
    }
    return std::make_tuple(std::move(spatula), std::move(rotations));
}

std::tuple<double, data::Vec3>
PGOP::compute_symmetry(LocalNeighborhood& neighborhood, const double* R_ij, size_t group_idx) const
{
    auto opt = m_optimize->clone();
    while (!opt->terminate()) {
        neighborhood.rotate(opt->next_point());
        const auto particle_op
            = compute_pgop(neighborhood, std::span<const double>(R_ij, m_group_sizes[group_idx]));
        opt->record_objective(-particle_op);
    }
    const auto optimum = opt->get_optimum();
    // op value is negated to get the correct value, because optimization scheme is
    // minimization not maximization!
    return std::make_tuple(-optimum.second, optimum.first);
}

double PGOP::compute_pgop(LocalNeighborhood& neighborhood, const std::span<const double> R_ij) const
{
    if (m_mode == 0) {
        if (neighborhood.constantSigmas()) {
            return computes::compute_pgop_gaussian_fast(neighborhood, R_ij);
        }
        return computes::compute_pgop_gaussian(neighborhood, R_ij);
    } else {
        if (neighborhood.constantSigmas()) {
            return computes::compute_pgop_fisher_fast(neighborhood, R_ij);
        }
        return computes::compute_pgop_fisher(neighborhood, R_ij);
    }
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
