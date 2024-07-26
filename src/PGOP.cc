#include <cmath>
#include <iterator>
#include <string>

#include "BondOrder.h"
#include "PGOP.h"
#include "util/Threads.h"
#include <nanobind/stl/shared_ptr.h>

namespace pgop {

Neighborhoods::Neighborhoods(size_t N,
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

LocalNeighborhood Neighborhoods::getNeighborhood(size_t i) const
{
    const size_t start {m_neighbor_offsets[i]}, end {m_neighbor_offsets[i + 1]};
    return LocalNeighborhood(
        util::normalize_distances(m_distances, std::make_pair(3 * start, 3 * end)),
        std::vector(m_weights + start, m_weights + end));
}

std::vector<data::Vec3> Neighborhoods::getNormalizedDistances(size_t i) const
{
    const size_t start {3 * m_neighbor_offsets[i]}, end {3 * m_neighbor_offsets[i + 1]};
    return util::normalize_distances(m_distances, std::make_pair(start, end));
}

std::vector<double> Neighborhoods::getWeights(size_t i) const
{
    const size_t start {m_neighbor_offsets[i]}, end {m_neighbor_offsets[i + 1]};
    return std::vector(m_weights + start, m_weights + end);
}

int Neighborhoods::getNeighborCount(size_t i) const
{
    return m_neighbor_counts[i];
}

LocalNeighborhood::LocalNeighborhood(std::vector<data::Vec3>&& positions_,
                                     std::vector<double>&& weights_)
    : positions(positions_), weights(weights_), rotated_positions(positions)
{
}

void LocalNeighborhood::rotate(const data::Vec3& v)
{
    const auto R = util::to_rotation_matrix(v);
    util::rotate_matrix(positions.cbegin(), positions.cend(), rotated_positions.begin(), R);
}

template<typename distribution_type>
PGOP<distribution_type>::PGOP(const nb::ndarray<std::complex<double>, nb::ndim<2>> D_ij,
                              std::shared_ptr<optimize::Optimizer>& optimizer,
                              typename distribution_type::param_type distribution_params)
    : m_distribution(distribution_params), m_n_symmetries(D_ij.shape(0)), m_Dij(),
      m_optimize(optimizer)
{
    m_Dij.reserve(m_n_symmetries);
    const std::complex<double>* u_D_ij = static_cast<const std::complex<double>*>(D_ij.data());
    const size_t n_mlms = D_ij.shape(1);
    for (size_t i {0}; i < m_n_symmetries; ++i) {
        m_Dij.emplace_back(
            std::vector<std::complex<double>>(u_D_ij + i * n_mlms, u_D_ij + (i + 1) * n_mlms));
    }
}

// TODO there is also a bug with self-neighbors.
template<typename distribution_type>
void PGOP<distribution_type>::compute(const nb::ndarray<double> distances,
                                      const nb::ndarray<double> weights,
                                      const nb::ndarray<int> num_neighbors,
                                      const unsigned int m,
                                      const nb::ndarray<std::complex<double>> ylms,
                                      const nb::ndarray<double> quad_positions,
                                      const nb::ndarray<double> quad_weights) const
{
    const auto qlm_eval = util::QlmEval(m, quad_positions, quad_weights, ylms);
    const auto neighborhoods = Neighborhoods(num_neighbors.size(),
                                             num_neighbors.data(),
                                             weights.data(),
                                             distances.data());
    const size_t N_particles = num_neighbors.size();
    // TODO FIX
    // reserve N_particles, m_n_symmetries for m_pgop_values
    m_pgop_values->reserve(N_particles);
    // reserve N_particles, m_n_symmetries, 4 for m_rotations
    m_optimal_rotations->reserve(N_particles);
    const auto loop_func = [&neighborhoods, &qlm_eval, this](const size_t start,
                                                             const size_t stop) {
        auto qlm_buf = util::QlmBuf(qlm_eval.getNlm());
        for (size_t i = start; i < stop; ++i) {
            if (neighborhoods.getNeighborCount(i) == 0) {
                for (size_t j {0}; j < m_n_symmetries; ++j) {
                    (*m_pgop_values)[i][j] = 0;
                    (*m_optimal_rotations)[i][j][0] = 1;
                    (*m_optimal_rotations)[i][j][1] = 0;
                    (*m_optimal_rotations)[i][j][2] = 0;
                    (*m_optimal_rotations)[i][j][3] = 0;
                }
                continue;
            }
            auto neighborhood = neighborhoods.getNeighborhood(i);
            const auto particle_op_rot = this->compute_particle(neighborhood, qlm_eval, qlm_buf);
            const auto& values = std::get<0>(particle_op_rot);
            const auto& rots = std::get<1>(particle_op_rot);
            for (size_t j {0}; j < m_n_symmetries; ++j) {
                (*m_pgop_values)[i][j] = values[j];
                (*m_optimal_rotations)[i][j][0] = rots[j].w;
                (*m_optimal_rotations)[i][j][1] = rots[j].x;
                (*m_optimal_rotations)[i][j][2] = rots[j].y;
                (*m_optimal_rotations)[i][j][3] = rots[j].z;
            }
        }
    };
    execute_func(loop_func, N_particles);
}

template<typename distribution_type>
void PGOP<distribution_type>::refine(const nb::ndarray<double> distances,
                                     const nb::ndarray<double> weights,
                                     const nb::ndarray<int> num_neighbors,
                                     const unsigned int m,
                                     const nb::ndarray<std::complex<double>> ylms,
                                     const nb::ndarray<double> quad_positions,
                                     const nb::ndarray<double> quad_weights) const
{
    const auto qlm_eval = util::QlmEval(m, quad_positions, quad_weights, ylms);
    const auto neighborhoods = Neighborhoods(num_neighbors.size(),
                                             num_neighbors.data(),
                                             weights.data(),
                                             distances.data());
    const size_t N_particles = num_neighbors.size();
    const auto loop_func
        = [&neighborhoods, &qlm_eval, this](const size_t start, const size_t stop) {
              auto qlm_buf = util::QlmBuf(qlm_eval.getNlm());
              for (size_t i = start; i < stop; ++i) {
                  if (neighborhoods.getNeighborCount(i) == 0) {
                      for (size_t j {0}; j < m_n_symmetries; ++j) {
                          (*m_pgop_values)[i][j] = 0;
                      }
                      continue;
                  }
                  auto neighborhood = neighborhoods.getNeighborhood(i);
                  for (size_t j {0}; j < m_n_symmetries; ++j) {
                      const auto& rot_array = (*m_optimal_rotations)[i][j];
                      const auto rot
                          = data::Quaternion(rot_array[0], rot_array[1], rot_array[2], rot_array[3])
                                .to_axis_angle_3D();
                      neighborhood.rotate(rot);
                      (*m_pgop_values)[i][j]
                          = this->compute_pgop(neighborhood, m_Dij[j], qlm_eval, qlm_buf);
                  }
              }
          };
    execute_func(loop_func, N_particles);
}

template<typename distribution_type>
std::tuple<std::vector<double>, std::vector<data::Quaternion>>
PGOP<distribution_type>::compute_particle(LocalNeighborhood& neighborhood,
                                          const util::QlmEval& qlm_eval,
                                          util::QlmBuf& qlm_buf) const
{
    auto pgop = std::vector<double>();
    auto rotations = std::vector<data::Quaternion>();
    pgop.reserve(m_Dij.size());
    rotations.reserve(m_Dij.size());
    for (const auto& D_ij : m_Dij) {
        const auto result = compute_symmetry(neighborhood, D_ij, qlm_eval, qlm_buf);
        pgop.emplace_back(std::get<0>(result));
        rotations.emplace_back(std::get<1>(result));
    }
    return std::make_tuple(std::move(pgop), std::move(rotations));
}

template<typename distribution_type>
std::tuple<double, data::Quaternion>
PGOP<distribution_type>::compute_symmetry(LocalNeighborhood& neighborhood,
                                          const std::vector<std::complex<double>>& D_ij,
                                          const util::QlmEval& qlm_eval,
                                          util::QlmBuf& qlm_buf) const
{
    auto opt = m_optimize->clone();
    while (!opt->terminate()) {
        neighborhood.rotate(opt->next_point());
        const auto particle_op = compute_pgop(neighborhood, D_ij, qlm_eval, qlm_buf);
        opt->record_objective(-particle_op);
    }
    // TODO currently optimum.first can be empty resulting in a SEGFAULT. This only happens in badly
    // formed arguments (particles with no neighbors), but can occur.
    const auto optimum = opt->get_optimum();
    return std::make_tuple(-optimum.second, optimum.first);
}

template<typename distribution_type>
double PGOP<distribution_type>::compute_pgop(LocalNeighborhood& neighborhood,
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
void PGOP<distribution_type>::execute_func(std::function<void(size_t, size_t)> func, size_t N) const
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

template class PGOP<UniformDistribution>;
template class PGOP<FisherDistribution>;

template<typename distribution_type> void export_pgop_class(nb::module_& m, const std::string& name)
{
    nb::class_<PGOP<distribution_type>>(m, name.c_str())
        .def(nb::init<const nb::ndarray<std::complex<double>, nb::ndim<2>>,
                      std::shared_ptr<optimize::Optimizer>&,
                      typename distribution_type::param_type>())
        .def("compute", &PGOP<distribution_type>::compute)
        .def("refine", &PGOP<distribution_type>::refine)
        .def("get_pgop_values", &PGOP<distribution_type>::get_pgop_values)
        .def("get_rotations", &PGOP<distribution_type>::get_rotations);
}

void export_pgop(nb::module_& m)
{
    export_pgop_class<UniformDistribution>(m, "PGOPUniform");
    export_pgop_class<FisherDistribution>(m, "PGOPFisher");
}
} // End namespace pgop
