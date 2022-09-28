#include <cmath>
#include <iterator>
#include <string>

#include "BondOrder.h"
#include "PGOP.h"
#include "optimize/Optimize.h"
#include "util/Threads.h"

namespace pgop {
template<typename distribution_type>
PGOP<distribution_type>::PGOP(unsigned int max_l,
                              const py::array_t<std::complex<double>> D_ij,
                              std::shared_ptr<optimize::Optimizer>& optimizer,
                              typename distribution_type::param_type distribution_params)
    : m_distribution(distribution_params), m_max_l(max_l), m_n_symmetries(D_ij.shape(0)), m_Dij(),
      m_optimize(optimizer)
{
    m_Dij.reserve(m_n_symmetries);
    const auto u_D_ij = D_ij.unchecked<2>();
    const size_t n_mlms = D_ij.shape(1);
    for (size_t i {0}; i < m_n_symmetries; ++i) {
        m_Dij.emplace_back(
            std::vector<std::complex<double>>(u_D_ij.data(i, 0), u_D_ij.data(i, 0) + n_mlms));
    }
}

// TODO There is a memory leak somewhere down this path. Not necessarily unaccessable but this will
// continue to eat memory until the system runs out if running for a long time.
// TODO there is also a bug with self-neighbors.
template<typename distribution_type>
py::tuple PGOP<distribution_type>::compute(const py::array_t<double> distances,
                                           const py::array_t<double> weights,
                                           const py::array_t<int> num_neighbors,
                                           const unsigned int m,
                                           const py::array_t<std::complex<double>> ylms,
                                           const py::array_t<double> quad_positions,
                                           const py::array_t<double> quad_weights) const
{
    const size_t N_particles = num_neighbors.size();
    const auto qlm_eval = util::QlmEval(m, quad_positions, quad_weights, ylms);
    const auto* neigh_count_ptr = static_cast<const int*>(num_neighbors.data(0));
    const auto* distances_ptr = static_cast<const double*>(distances.data(0));
    const auto* weights_ptr = static_cast<const double*>(weights.data(0));
    const auto op_shape = std::vector<size_t>({N_particles, m_n_symmetries});
    auto op = py::array_t<double>(op_shape);
    auto u_op = op.mutable_unchecked<2>();
    auto rotations_shape = op_shape;
    rotations_shape.push_back(4);
    auto rotations = py::array_t<double>(rotations_shape);
    auto u_rotations = rotations.mutable_unchecked<3>();
    auto distance_offsets = std::vector<size_t>();
    distance_offsets.reserve(N_particles + 1);
    distance_offsets.emplace_back(0);
    std::partial_sum(neigh_count_ptr,
                     neigh_count_ptr + N_particles,
                     std::back_inserter(distance_offsets));
    const auto loop_func = [&u_op,
                            &u_rotations,
                            &distance_offsets,
                            &neigh_count_ptr,
                            &qlm_eval,
                            &distances_ptr,
                            &weights_ptr,
                            this](const size_t start, const size_t stop) {
        auto qlm_buf = util::QlmBuf(qlm_eval.getNlm());
        for (size_t i = start; i < stop; ++i) {
            if (neigh_count_ptr[i] == 0) {
                continue;
            }
            const auto particle_op_rot
                = this->compute_particle(util::normalize_distances(std::ranges::subrange(
                                             distances_ptr + 3 * distance_offsets[i],
                                             distances_ptr + 3 * distance_offsets[i + 1])),
                                         std::vector<double>(weights_ptr + distance_offsets[i],
                                                             weights_ptr + distance_offsets[i + 1]),
                                         qlm_eval,
                                         qlm_buf);
            const auto particle_op = std::get<0>(particle_op_rot);
            const auto particle_rotations = std::get<1>(particle_op_rot);
            for (size_t j {0}; j < particle_op.size(); ++j) {
                u_op(i, j) = particle_op[j];
                u_rotations(i, j, 0) = particle_rotations[j].w;
                u_rotations(i, j, 1) = particle_rotations[j].x;
                u_rotations(i, j, 2) = particle_rotations[j].y;
                u_rotations(i, j, 3) = particle_rotations[j].z;
            }
        }
    };
    bool serial = false;
    // Enable profiling through serial mode.
    if (serial) {
        util::ThreadPool::get().serial_compute<void, size_t>(0, N_particles, loop_func);
    } else {
        auto& pool = util::ThreadPool::get().get_pool();
        pool.push_loop(0, N_particles, loop_func, 2 * pool.get_thread_count());
        pool.wait_for_tasks();
    }
    return py::make_tuple(op, rotations);
}

template<typename distribution_type>
std::tuple<std::vector<double>, std::vector<data::Quaternion>>
PGOP<distribution_type>::compute_particle(const std::vector<data::Vec3>& positions,
                                          const std::vector<double>& weights,
                                          const util::QlmEval& qlm_eval,
                                          util::QlmBuf& qlm_buf) const
{
    auto rotated_dist = std::vector<data::Vec3>(positions);
    auto pgop = std::vector<double>();
    auto rotations = std::vector<data::Quaternion>();
    pgop.reserve(m_Dij.size());
    rotations.reserve(m_Dij.size());
    for (const auto& D_ij : m_Dij) {
        const auto result
            = compute_symmetry(positions, weights, rotated_dist, D_ij, qlm_eval, qlm_buf);
        pgop.push_back(std::get<0>(result));
        rotations.push_back(std::get<1>(result));
    }
    return std::make_tuple(std::move(pgop), std::move(rotations));
}

template<typename distribution_type>
std::tuple<double, data::Quaternion>
PGOP<distribution_type>::compute_symmetry(const std::vector<data::Vec3>& positions,
                                          const std::vector<double>& weights,
                                          std::vector<data::Vec3>& rotated_distances_buf,
                                          const std::vector<std::complex<double>>& D_ij,
                                          const util::QlmEval& qlm_eval,
                                          util::QlmBuf& qlm_buf) const
{
    // Optimize over the 4D unit sphere which has a bijective mapping from unit quaternions to the
    // hypersphere's surface.
    const std::vector<double> opt_min_bounds {-M_PI, -M_PI_2, -M_PI_2};
    const std::vector<double> opt_max_bounds {M_PI, M_PI_2, M_PI_2};
    auto opt = m_optimize->clone();
    while (!opt->terminate()) {
        const auto hsphere_pos = opt->next_point();
        const auto particle_op = compute_pgop(hsphere_pos,
                                              positions,
                                              weights,
                                              rotated_distances_buf,
                                              D_ij,
                                              qlm_eval,
                                              qlm_buf);
        opt->record_objective(-particle_op);
    }
    // TODO currently optimum.first can be empty resulting in a SEGFAULT. This only happens in badly
    // formed arguments (particles with no neighbors), but can occur.
    const auto optimum = opt->get_optimum();
    return std::make_tuple(
        -optimum.second,
        util::quat_from_hypersphere(optimum.first[0], optimum.first[1], optimum.first[2]));
}

template<typename distribution_type>
double PGOP<distribution_type>::compute_pgop(const std::vector<double>& hsphere_pos,
                                             const std::vector<data::Vec3>& positions,
                                             const std::vector<double>& weights,
                                             std::vector<data::Vec3>& rotated_positions,
                                             const std::vector<std::complex<double>>& D_ij,
                                             const util::QlmEval& qlm_eval,
                                             util::QlmBuf& qlm_buf) const
{
    const auto R = data::quat_from_hypersphere(hsphere_pos[0], hsphere_pos[1], hsphere_pos[2])
                       .to_rotation_matrix();
    util::rotate_matrix(positions.begin(), positions.end(), rotated_positions.begin(), R);
    const auto bond_order
        = BondOrder<distribution_type>(m_distribution, rotated_positions, weights);
    // compute spherical harmonic values in-place (qlm_buf.qlms)
    qlm_eval.eval<distribution_type>(bond_order, qlm_buf.qlms);
    util::symmetrize_qlm(qlm_buf.qlms, D_ij, qlm_buf.sym_qlms, m_max_l);
    return util::covariance(qlm_buf.qlms, qlm_buf.sym_qlms);
}

template class PGOP<UniformDistribution>;
template class PGOP<FisherDistribution>;

template<typename distribution_type> void export_pgop_class(py::module& m, const std::string& name)
{
    py::class_<PGOP<distribution_type>>(m, name.c_str())
        .def(py::init<unsigned int,
                      const py::array_t<std::complex<double>>,
                      std::shared_ptr<optimize::Optimizer>&,
                      typename distribution_type::param_type>())
        .def("compute", &PGOP<distribution_type>::compute);
}

void export_pgop(py::module& m)
{
    export_pgop_class<UniformDistribution>(m, "PGOPUniform");
    export_pgop_class<FisherDistribution>(m, "PGOPFisher");
}
} // End namespace pgop
