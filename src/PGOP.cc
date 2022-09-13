#include <cmath>
#include <iterator>
#include <string>

#include "BondOrder.h"
#include "Optimize.h"
#include "PGOP.h"
#include "Threads.h"
#include "Weijer.h"

template<typename distribution_type>
PGOP<distribution_type>::PGOP(unsigned int max_l,
                              const py::array_t<std::complex<double>> D_ij,
                              std::shared_ptr<Optimizer>& optimizer,
                              typename distribution_type::param_type distribution_params)
    : m_distribution_params(distribution_params), m_max_l(max_l), m_n_symmetries(D_ij.shape(0)),
      m_Dij(), m_optimize(optimizer)
{
    m_Dij.reserve(m_n_symmetries);
    const auto u_D_ij = D_ij.unchecked<2>();
    const size_t n_mlms = D_ij.shape(1);
    for (size_t i {0}; i < m_n_symmetries; ++i) {
        m_Dij.emplace_back(
            std::vector<std::complex<double>>(u_D_ij.data(i, 0), u_D_ij.data(i, 0) + n_mlms));
    }
}

template<typename distribution_type>
py::tuple PGOP<distribution_type>::compute(const py::array_t<double> distances,
                                           const py::array_t<int> num_neighbors,
                                           const unsigned int m,
                                           const py::array_t<std::complex<double>> ylms,
                                           const py::array_t<double> quad_positions,
                                           const py::array_t<double> quad_weights) const
{
    const size_t N_particles = num_neighbors.size();
    const auto qlm_eval = QlmEval(m, quad_positions, quad_weights, ylms);
    const auto* neigh_count_ptr = static_cast<const int*>(num_neighbors.data(0));
    const auto normed_distances = normalize_distances(distances);
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
    const auto dist_begin = normed_distances.cbegin();
    const auto loop_func =
        [&u_op, &u_rotations, &distance_offsets, &qlm_eval, &dist_begin, this](const size_t start,
                                                                               const size_t stop) {
            for (size_t i = start; i < stop; ++i) {
                const auto particle_op_rot
                    = this->compute_particle(dist_begin + distance_offsets[i],
                                             dist_begin + distance_offsets[i + 1],
                                             qlm_eval);
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
        ThreadPool::get().serial_compute<void, size_t>(0, N_particles, loop_func);
    } else {
        auto& pool = ThreadPool::get().get_pool();
        pool.push_loop(0, N_particles, loop_func, 2 * pool.get_thread_count());
        pool.wait_for_tasks();
    }
    return py::make_tuple(op, rotations);
}

template<typename distribution_type>
std::tuple<std::vector<double>, std::vector<Quaternion>>
PGOP<distribution_type>::compute_particle(const std::vector<Vec3>::const_iterator& position_begin,
                                          const std::vector<Vec3>::const_iterator& position_end,
                                          const QlmEval& qlm_eval) const
{
    auto rotated_dist = std::vector<Vec3>(std::distance(position_begin, position_end));
    auto sym_qlm_buf = std::vector<std::complex<double>>();
    sym_qlm_buf.reserve(qlm_eval.getNlm());
    auto pgop = std::vector<double>();
    auto rotations = std::vector<Quaternion>();
    pgop.reserve(m_Dij.size());
    rotations.reserve(m_Dij.size());
    for (const auto& D_ij : m_Dij) {
        const auto result = compute_symmetry(position_begin,
                                             position_end,
                                             rotated_dist,
                                             D_ij,
                                             sym_qlm_buf,
                                             qlm_eval);
        pgop.push_back(std::get<0>(result));
        rotations.push_back(std::get<1>(result));
    }
    return std::make_tuple(std::move(pgop), std::move(rotations));
}

template<typename distribution_type>
std::tuple<double, Quaternion>
PGOP<distribution_type>::compute_symmetry(const std::vector<Vec3>::const_iterator& position_begin,
                                          const std::vector<Vec3>::const_iterator& position_end,
                                          std::vector<Vec3>& rotated_distances_buf,
                                          const std::vector<std::complex<double>>& D_ij,
                                          std::vector<std::complex<double>>& sym_qlm_buf,
                                          const QlmEval& qlm_eval) const
{
    // Optimize over the 4D unit sphere which has a bijective mapping from unit quaternions to the
    // hypersphere's surface.
    const std::vector<double> opt_min_bounds {-M_PI, -M_PI_2, -M_PI_2};
    const std::vector<double> opt_max_bounds {M_PI, M_PI_2, M_PI_2};
    auto opt = m_optimize->clone();
    while (!opt->terminate()) {
        const auto hsphere_pos = opt->next_point();
        const auto particle_op = compute_pgop(hsphere_pos,
                                              position_begin,
                                              position_end,
                                              rotated_distances_buf,
                                              D_ij,
                                              sym_qlm_buf,
                                              qlm_eval);
        opt->record_objective(-particle_op);
    }
    const auto best_rotation = opt->get_optimum().first;
    return std::make_tuple(
        compute_pgop(best_rotation,
                     position_begin,
                     position_end,
                     rotated_distances_buf,
                     D_ij,
                     sym_qlm_buf,
                     qlm_eval),
        quat_from_hypersphere(best_rotation[0], best_rotation[1], best_rotation[2]));
}

template<typename distribution_type>
double
PGOP<distribution_type>::compute_pgop(const std::vector<double>& hsphere_pos,
                                      const std::vector<Vec3>::const_iterator& position_begin,
                                      const std::vector<Vec3>::const_iterator& position_end,
                                      std::vector<Vec3>& rotated_positions,
                                      const std::vector<std::complex<double>>& D_ij,
                                      std::vector<std::complex<double>>& sym_qlm_buf,
                                      const QlmEval& qlm_eval) const
{
    const auto R = quat_from_hypersphere(hsphere_pos[0], hsphere_pos[1], hsphere_pos[2])
                       .to_rotation_matrix();
    rotate_matrix(position_begin, position_end, rotated_positions.begin(), R);
    const auto bond_order = BondOrder<distribution_type>(getDistribution(), rotated_positions);
    const auto qlms = qlm_eval.eval<distribution_type>(bond_order);
    symmetrize_qlm(qlms, D_ij, sym_qlm_buf, m_max_l);
    return covariance(qlms, sym_qlm_buf);
}

template<typename distribution_type>
std::vector<std::vector<double>> PGOP<distribution_type>::getDefaultRotations() const
{
    auto rotations = std::vector<std::vector<double>>();
    auto phis = linspace(-M_PI, M_PI_2, 10, false);
    auto thetas = linspace(-M_PI_2, M_PI_2, 5, true);
    auto psis = linspace(-M_PI_2, M_PI_2, 5, true);
    rotations.reserve(phis.size() * thetas.size() * psis.size());
    for (const auto& phi : phis) {
        for (const auto& theta : thetas) {
            for (const auto& psi : psis) {
                rotations.push_back(std::initializer_list<double> {phi, theta, psi});
            }
        }
    }
    return rotations;
}

template<typename distribution_type>
std::vector<std::vector<double>>
PGOP<distribution_type>::getInitialSimplex(const std::vector<double>& center) const
{
    const double delta_phi {M_PI / 8}, delta_theta {M_PI / 16}, delta_psi {M_PI / 16};
    const double b = delta_psi / std::sqrt(2);
    const double b_plus_z {b + center[2]}, z_minus_b {center[2] - b};
    return std::vector<std::vector<double>> {
        std::initializer_list<double> {center[0] + delta_phi, center[1], z_minus_b},
        std::initializer_list<double> {center[0] - delta_phi, center[1], z_minus_b},
        std::initializer_list<double> {center[0], center[1] + delta_theta, b_plus_z},
        std::initializer_list<double> {center[0], center[1] - delta_theta, b_plus_z}};
}

template<typename distribution_type>
distribution_type PGOP<distribution_type>::getDistribution() const
{
    return distribution_type(m_distribution_params);
}

template class PGOP<UniformDistribution>;
template class PGOP<FisherDistribution>;

template<typename distribution_type> void export_pgop_class(py::module& m, const std::string& name)
{
    py::class_<PGOP<distribution_type>>(m, name.c_str())
        .def(py::init<unsigned int,
                      const py::array_t<std::complex<double>>,
                      std::shared_ptr<Optimizer>&,
                      typename distribution_type::param_type>())
        .def("compute", &PGOP<distribution_type>::compute);
}

void export_pgop(py::module& m)
{
    export_pgop_class<UniformDistribution>(m, "PGOPUniform");
    export_pgop_class<FisherDistribution>(m, "PGOPFisher");
}
