// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <cmath>
#include <iterator>
#include <string>

#include "BOOSOP.h"
#include "BondOrder.h"
#include "util/Threads.h"

namespace spatula {

NeighborhoodBOOs::NeighborhoodBOOs(size_t N,
                                   const int* neighbor_counts,
                                   const double* weights,
                                   const double* distance)
    : m_N {N}, m_neighbor_counts {neighbor_counts}, m_distances {distance}, m_weights {weights},
      m_neighbor_offsets()
{
    m_neighbor_offsets.reserve(m_N + 1);
    m_neighbor_offsets.emplace_back(0);
    std::partial_sum(m_neighbor_counts,
                     m_neighbor_counts + m_N,
                     std::back_inserter(m_neighbor_offsets));
}

LocalNeighborhoodBOOBOO NeighborhoodBOOs::getNeighborhoodBOO(size_t i) const
{
    const size_t start {m_neighbor_offsets[i]}, end {m_neighbor_offsets[i + 1]};
    return LocalNeighborhoodBOOBOO(
        util::normalize_distances(m_distances, std::make_pair(3 * start, 3 * end)),
        std::vector(m_weights + start, m_weights + end));
}

std::vector<data::Vec3> NeighborhoodBOOs::getNormalizedDistances(size_t i) const
{
    const size_t start {3 * m_neighbor_offsets[i]}, end {3 * m_neighbor_offsets[i + 1]};
    return util::normalize_distances(m_distances, std::make_pair(start, end));
}

std::vector<double> NeighborhoodBOOs::getWeights(size_t i) const
{
    const size_t start {m_neighbor_offsets[i]}, end {m_neighbor_offsets[i + 1]};
    return std::vector(m_weights + start, m_weights + end);
}

int NeighborhoodBOOs::getNeighborCount(size_t i) const
{
    return m_neighbor_counts[i];
}

LocalNeighborhoodBOOBOO::LocalNeighborhoodBOOBOO(std::vector<data::Vec3>&& positions_,
                                                 std::vector<double>&& weights_)
    : positions(positions_), weights(weights_), rotated_positions(positions)
{
}

void LocalNeighborhoodBOOBOO::rotate(const data::Vec3& v)
{
    const auto R = util::to_rotation_matrix(v);
    util::rotate_matrix(positions.cbegin(), positions.cend(), rotated_positions.begin(), R);
}

BOOSOPStore::BOOSOPStore(size_t N_particles, size_t N_symmetries)
    : N_syms(N_symmetries),
      op(N_particles * N_symmetries),
      rotations(N_particles * N_symmetries * 4),
      m_N_particles(N_particles)
{
}

void BOOSOPStore::addOp(size_t i,
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

void BOOSOPStore::addNull(size_t i)
{
    for (size_t j {0}; j < N_syms; ++j) {
        op[i * N_syms + j] = 0;
        rotations[(i * N_syms + j) * 4 + 0] = 1;
        rotations[(i * N_syms + j) * 4 + 1] = 0;
        rotations[(i * N_syms + j) * 4 + 2] = 0;
        rotations[(i * N_syms + j) * 4 + 3] = 0;
    }
}

std::pair<std::vector<double>, std::vector<double>> BOOSOPStore::getArrays()
{
    return std::make_pair(op, rotations);
}

template<typename distribution_type>
BOOSOP<distribution_type>::BOOSOP(const std::vector<std::vector<std::complex<double>>>& D_ij,
                                  std::shared_ptr<optimize::Optimizer>& optimizer,
                                  typename distribution_type::param_type distribution_params)
    : m_distribution(distribution_params), m_n_symmetries(D_ij.size()), m_Dij(D_ij),
      m_optimize(optimizer)
{
}

// TODO there is also a bug with self-neighbors.
template<typename distribution_type>
std::pair<std::vector<double>, std::vector<double>>
BOOSOP<distribution_type>::compute(size_t N_points,
                                   const double* distances,
                                   const double* weights,
                                   const int* num_neighbors,
                                   const unsigned int m,
                                   const std::complex<double>* ylms,
                                   const double* quad_positions,
                                   const double* quad_weights) const
{
    const auto qlm_eval = util::QlmEval(m, quad_positions, quad_weights, ylms);
    const auto neighborhoods = NeighborhoodBOOs(N_points,
                                                num_neighbors,
                                                weights,
                                                distances);
    const size_t N_particles = N_points;
    auto op_store = BOOSOPStore(N_particles, m_n_symmetries);
    const auto loop_func = [&op_store, &neighborhoods, &qlm_eval, this](const size_t start,
                                                                        const size_t stop) {
        auto qlm_buf = util::QlmBuf(qlm_eval.getNlm());
        for (size_t i = start; i < stop; ++i) {
            if (neighborhoods.getNeighborCount(i) == 0) {
                op_store.addNull(i);
                continue;
            }
            auto neighborhood = neighborhoods.getNeighborhoodBOO(i);
            const auto particle_op_rot = this->compute_particle(neighborhood, qlm_eval, qlm_buf);
            op_store.addOp(i, particle_op_rot);
        }
    };
    execute_func(loop_func, N_particles);
    return op_store.getArrays();
}

template<typename distribution_type>
std::vector<double> BOOSOP<distribution_type>::refine(size_t N_points,
                                                      const double* distances,
                                                      const double* rotations,
                                                      const double* weights,
                                                      const int* num_neighbors,
                                                                                                            const unsigned int m,
                                                      const std::complex<double>* ylms,
                                                      const double* quad_positions,
                                                      const double* quad_weights) const
{
    const auto qlm_eval = util::QlmEval(m, quad_positions, quad_weights, ylms);
    const auto neighborhoods = NeighborhoodBOOs(N_points,
                                                num_neighbors,
                                                weights,
                                                distances);
    const size_t N_particles = N_points;
    std::vector<double> op_store(N_particles * m_n_symmetries);
    const auto loop_func
        = [&op_store, &rotations, &neighborhoods, &qlm_eval, this, N_points](const size_t start,
                                                                       const size_t stop) {
              auto qlm_buf = util::QlmBuf(qlm_eval.getNlm());
              for (size_t i = start; i < stop; ++i) {
                  if (neighborhoods.getNeighborCount(i) == 0) {
                      for (size_t j {0}; j < m_n_symmetries; ++j) {
                          op_store[i * m_n_symmetries + j] = 0;
                      }
                      continue;
                  }
                  auto neighborhood = neighborhoods.getNeighborhoodBOO(i);
                  for (size_t j {0}; j < m_n_symmetries; ++j) {
                      const auto rot = data::Quaternion(rotations[(i * m_n_symmetries + j) * 4 + 0],
                                                        rotations[(i * m_n_symmetries + j) * 4 + 1],
                                                        rotations[(i * m_n_symmetries + j) * 4 + 2],
                                                        rotations[(i * m_n_symmetries + j) * 4 + 3])
                                           .to_axis_angle_3D();
                      neighborhood.rotate(rot);
                      op_store[i * m_n_symmetries + j]
                          = this->compute_BOOSOP(neighborhood, m_Dij[j], qlm_eval, qlm_buf);
                  }
              }
          };
    execute_func(loop_func, N_particles);
    return op_store;
}

template<typename distribution_type>
std::tuple<std::vector<double>, std::vector<data::Quaternion>>
BOOSOP<distribution_type>::compute_particle(LocalNeighborhoodBOOBOO& neighborhood,
                                            const util::QlmEval& qlm_eval,
                                            util::QlmBuf& qlm_buf) const
{
    auto BOOSOP = std::vector<double>();
    auto rotations = std::vector<data::Quaternion>();
    BOOSOP.reserve(m_Dij.size());
    rotations.reserve(m_Dij.size());
    for (const auto& D_ij : m_Dij) {
        const auto result = compute_symmetry(neighborhood, D_ij, qlm_eval, qlm_buf);
        BOOSOP.emplace_back(std::get<0>(result));
        rotations.emplace_back(std::get<1>(result));
    }
    return std::make_tuple(std::move(BOOSOP), std::move(rotations));
}

template<typename distribution_type>
std::tuple<double, data::Quaternion>
BOOSOP<distribution_type>::compute_symmetry(LocalNeighborhoodBOOBOO& neighborhood,
                                            const std::vector<std::complex<double>>& D_ij,
                                            const util::QlmEval& qlm_eval,
                                            util::QlmBuf& qlm_buf) const
{
    auto opt = m_optimize->clone();
    while (!opt->terminate()) {
        neighborhood.rotate(opt->next_point());
        const auto particle_op = compute_BOOSOP(neighborhood, D_ij, qlm_eval, qlm_buf);
        opt->record_objective(-particle_op);
    }
    // TODO currently optimum.first can be empty resulting in a SEGFAULT. This only happens in badly
    // formed arguments (particles with no neighbors), but can occur.
    const auto optimum = opt->get_optimum();
    return std::make_tuple(-optimum.second, optimum.first);
}

template<typename distribution_type>
double BOOSOP<distribution_type>::compute_BOOSOP(LocalNeighborhoodBOOBOO& neighborhood,
                                                 const std::vector<std::complex<double>>& D_ij,
                                                 const util::QlmEval& qlm_eval,
                                                 util::QlmBuf& qlm_buf) const
{
    const auto bond_order = BondOrder<distribution_type>(m_distribution,
                                                         neighborhood.rotated_positions,
                                                         neighborhood.weights);
    // compute spherical harmonic values in-place (qlm_buf.qlms)
    qlm_eval.eval<distribution_type>(bond_order, qlm_buf.qlms);
    util::symmetrize_qlm(qlm_buf.qlms, D_ij, qlm_buf.sym_qlms, qlm_eval.getMaxL());
    return util::covariance(qlm_buf.qlms, qlm_buf.sym_qlms);
}

template<typename distribution_type>
void BOOSOP<distribution_type>::execute_func(std::function<void(size_t, size_t)> func,
                                             size_t N) const
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

template class BOOSOP<UniformDistribution>;
template class BOOSOP<FisherDistribution>;

} // namespace spatula
