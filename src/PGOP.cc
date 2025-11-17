// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <cmath>
#include <iterator>
#include <string>

#include "PGOP.h"
#include "util/Threads.h"

namespace spatula {

Neighborhoods::Neighborhoods(size_t N,
                             const int* neighbor_counts,
                             const double* weights,
                             const double* distance,
                             const double* sigmas)
    : m_N {N}, m_neighbor_counts {neighbor_counts}, m_distances {distance}, m_weights {weights},
      m_sigmas {sigmas}, m_neighbor_offsets()
{
    m_neighbor_offsets.reserve(m_N + 1);
    m_neighbor_offsets.emplace_back(0);
    std::partial_sum(m_neighbor_counts,
                     m_neighbor_counts + m_N,
                     std::back_inserter(m_neighbor_offsets));
}

LocalNeighborhood Neighborhoods::getNeighborhood(size_t i) const
{
    const size_t start {m_neighbor_offsets[i]}, end {m_neighbor_offsets[i + 1]};

    // Create a vector of Vec3 to store the positions (3 coordinates for each Vec3)
    std::vector<data::Vec3> neighborhood_positions;
    neighborhood_positions.reserve(end - start);

    for (size_t j = start; j < end; ++j) {
        // Each Vec3 contains 3 consecutive elements from m_distances
        neighborhood_positions.emplace_back(
            data::Vec3 {m_distances[3 * j], m_distances[3 * j + 1], m_distances[3 * j + 2]});
    }

    return LocalNeighborhood(std::move(neighborhood_positions),
                             std::vector(m_weights + start, m_weights + end),
                             std::vector(m_sigmas + start, m_sigmas + end));
}

std::vector<double> Neighborhoods::getWeights(size_t i) const
{
    const size_t start {m_neighbor_offsets[i]}, end {m_neighbor_offsets[i + 1]};
    return std::vector(m_weights + start, m_weights + end);
}

std::vector<double> Neighborhoods::getSigmas(size_t i) const
{
    const size_t start {m_neighbor_offsets[i]}, end {m_neighbor_offsets[i + 1]};
    return std::vector(m_sigmas + start, m_sigmas + end);
}

int Neighborhoods::getNeighborCount(size_t i) const
{
    return m_neighbor_counts[i];
}

LocalNeighborhood::LocalNeighborhood(std::vector<data::Vec3>&& positions_,
                                     std::vector<double>&& weights_,
                                     std::vector<double>&& sigmas_)
    : positions(positions_), weights(weights_), sigmas(sigmas_), rotated_positions(positions)
{
}

void LocalNeighborhood::rotate(const data::Vec3& v)
{
    const auto R = util::to_rotation_matrix(v);
    util::rotate_matrix(positions.cbegin(), positions.cend(), rotated_positions.begin(), R);
}

PGOPStore::PGOPStore(size_t N_particles, size_t N_symmetries)
    : N_syms(N_symmetries),
      op(N_particles * N_symmetries),
      rotations(N_particles * N_symmetries * 4),
      m_N_particles(N_particles),
      m_N_symmetries(N_symmetries)
{
}

void PGOPStore::addOp(size_t i,
                      const std::tuple<std::vector<double>, std::vector<data::Quaternion>>& op_)
{
    const auto& values = std::get<0>(op_);
    const auto& rots = std::get<1>(op_);
    for (size_t j {0}; j < N_syms; ++j) {
        op[i * N_syms + j] = values[j];
        rotations[(i * N_syms + j) * 4 + 0] = rots[j].w;
        rotations[(i * N_syms + j) * 4 + 1] = rots[j].x;
        rotations[(i * N_syms + j) * 4 + 2] = rots[j].y;
        rotations[(i * N_syms + j) * 4 + 3] = rots[j].z;
    }
}

void PGOPStore::addNull(size_t i)
{
    for (size_t j {0}; j < N_syms; ++j) {
        op[i * N_syms + j] = std::numeric_limits<double>::quiet_NaN(); // Set NaN
        rotations[(i * N_syms + j) * 4 + 0] = 1;
        rotations[(i * N_syms + j) * 4 + 1] = 0;
        rotations[(i * N_syms + j) * 4 + 2] = 0;
        rotations[(i * N_syms + j) * 4 + 3] = 0;
    }
}

std::pair<std::vector<double>, std::vector<double>> PGOPStore::getArrays()
{
    return std::make_pair(op, rotations);
}

PGOP::PGOP(const std::vector<std::vector<double>>& R_ij,
           std::shared_ptr<optimize::Optimizer>& optimizer,
           const unsigned int mode,
           bool compute_per_operator)
    : m_n_symmetries(R_ij.size()), m_Rij(R_ij), m_optimize(optimizer), m_mode(mode),
      m_compute_per_operator(compute_per_operator)
{
}

std::pair<std::vector<double>, std::vector<double>> PGOP::compute(size_t N_points,
                                                                 const double* distances,
                                                                 const double* weights,
                                                                 const int* num_neighbors,
                                                                 const double* sigmas) const
{
    const auto neighborhoods = Neighborhoods(N_points,
                                             num_neighbors,
                                             weights,
                                             distances,
                                             sigmas);
    const size_t N_particles = N_points;
    auto total_number_of_op_to_store = m_n_symmetries;
    if (m_compute_per_operator) {
        for (const auto& R_ij : m_Rij) {
            total_number_of_op_to_store += R_ij.size() / 9;
        }
    }
    auto op_store = PGOPStore(N_particles, total_number_of_op_to_store);
    const auto loop_func
        = [&op_store, &neighborhoods, this](const size_t start, const size_t stop) {
              for (size_t i = start; i < stop; ++i) {
                  if (neighborhoods.getNeighborCount(i) == 0) {
                      op_store.addNull(i);
                      continue;
                  }
                  auto neighborhood = neighborhoods.getNeighborhood(i);
                  const auto particle_op_rot = this->compute_particle(neighborhood);
                  op_store.addOp(i, particle_op_rot);
              }
          };
    execute_func(loop_func, N_particles);
    return op_store.getArrays();
}

std::tuple<std::vector<double>, std::vector<data::Quaternion>>
PGOP::compute_particle(LocalNeighborhood& neighborhood_original) const
{
    auto spatula = std::vector<double>();
    auto rotations = std::vector<data::Quaternion>();
    spatula.reserve(m_Rij.size());
    rotations.reserve(m_Rij.size());
    for (const auto& R_ij : m_Rij) {
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
            for (size_t i = 0; i < R_ij.size(); i += 9) {
                const auto particle_operator_op
                    = compute_pgop(neighborhood,
                                   std::vector(R_ij.begin() + i, R_ij.begin() + i + 9));
                spatula.emplace_back(particle_operator_op);
                rotations.emplace_back(quat);
            }
        }
    }
    return std::make_tuple(std::move(spatula), std::move(rotations));
}

std::tuple<double, data::Vec3> PGOP::compute_symmetry(LocalNeighborhood& neighborhood,
                                                      const std::vector<double>& R_ij) const
{
    auto opt = m_optimize->clone();
    while (!opt->terminate()) {
        neighborhood.rotate(opt->next_point());
        const auto particle_op = compute_pgop(neighborhood, R_ij);
        opt->record_objective(-particle_op);
    }
    const auto optimum = opt->get_optimum();
    // op value is negated to get the correct value, because optimization scheme is
    // minimization not maximization!
    return std::make_tuple(-optimum.second, optimum.first);
}

inline double compute_Bhattacharyya_coefficient_gaussian(const data::Vec3& position,
                                                         const data::Vec3& symmetrized_position,
                                                         double sigma,
                                                         double sigma_symmetrized)
{
    // 1. compute the distance between the two vectors (symmetrized_position
    //    and positions[m])
    auto r_pos = symmetrized_position - position;
    auto sigmas_squared_summed = sigma * sigma + sigma_symmetrized * sigma_symmetrized;
    // 2. compute the gaussian overlap between the two points. Bhattacharyya coefficient
    //    is used.
    return std::pow((2 * sigma * sigma_symmetrized / sigmas_squared_summed), 3 / 2)
           * std::exp(-r_pos.dot(r_pos) / (4 * sigmas_squared_summed));
}

inline double compute_Bhattacharyya_coefficient_fisher(const data::Vec3& position,
                                                       const data::Vec3& symmetrized_position,
                                                       double kappa,
                                                       double kappa_symmetrized)
{
    auto position_norm = std::sqrt(position.dot(position));
    auto symmetrized_position_norm = std::sqrt(symmetrized_position.dot(symmetrized_position));
    // If position norm is zero vector means this point is at origin and contributes 1
    // to the overlap, check that with a small epsilon.
    if ((position_norm < 1e-10) && (symmetrized_position_norm < 1e-10)) {
        return 1;
    } else if ((position_norm < 1e-10) || (symmetrized_position_norm < 1e-10)) {
        return 0;
    }
    auto k1_sq = kappa * kappa;
    auto k2_sq = kappa_symmetrized * kappa_symmetrized;
    auto k1k2 = kappa * kappa_symmetrized;
    auto proj = position.dot(symmetrized_position) / (position_norm * symmetrized_position_norm);
    return 2 * std::sqrt(k1k2 / (std::sinh(kappa) * std::sinh(kappa_symmetrized)))
           * std::sinh((std::sqrt(k1_sq + k2_sq + 2 * k1k2 * proj)) / 2)
           / std::sqrt(k1_sq + k2_sq + 2 * k1k2 * proj);
}

double PGOP::compute_pgop(LocalNeighborhood& neighborhood, const std::vector<double>& R_ij) const
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
                    BC = compute_Bhattacharyya_coefficient_gaussian(positions[m],
                                                                    symmetrized_position,
                                                                    sigmas[j],
                                                                    sigmas[m]);
                } else {
                    BC = compute_Bhattacharyya_coefficient_fisher(positions[m],
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
