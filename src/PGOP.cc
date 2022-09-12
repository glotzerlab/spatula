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
                              std::unique_ptr<WeightedPNormBase> p_norm,
                              typename distribution_type::param_type distribution_params)
    : m_distribution_params(distribution_params), m_max_l(max_l), m_n_symmetries(D_ij.shape(0)),
      m_Dij(), m_p_norm(std::move(p_norm))
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
py::array_t<double> PGOP<distribution_type>::compute(const py::array_t<double> distances,
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
    auto distance_offsets = std::vector<size_t>();
    distance_offsets.reserve(N_particles + 1);
    distance_offsets.emplace_back(0);
    std::partial_sum(neigh_count_ptr,
                     neigh_count_ptr + N_particles,
                     std::back_inserter(distance_offsets));
    const auto dist_begin = normed_distances.cbegin();
    const auto loop_func
        = [&u_op, &normed_distances, &distance_offsets, &qlm_eval, &dist_begin, this](
              const size_t start,
              const size_t stop) {
              for (size_t i = start; i < stop; ++i) {
                  const auto particle_op
                      = this->compute_particle(dist_begin + distance_offsets[i],
                                               dist_begin + distance_offsets[i + 1],
                                               qlm_eval);
                  for (size_t j {0}; j < particle_op.size(); ++j) {
                      u_op(i, j) = particle_op[j];
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
    return op;
}

template<typename distribution_type>
std::vector<double>
PGOP<distribution_type>::compute_particle(const std::vector<Vec3>::const_iterator& position_begin,
                                          const std::vector<Vec3>::const_iterator& position_end,
                                          const QlmEval& qlm_eval) const
{
    const std::vector<double> opt_min_bounds {-M_PI, -M_PI_2, -M_PI};
    const std::vector<double> opt_max_bounds {M_PI, M_PI_2, M_PI};
    auto brute_opt = BruteForce(getDefaultRotations(), opt_min_bounds, opt_max_bounds);
    auto rotated_dist = std::vector<Vec3>(std::distance(position_begin, position_end));
    auto sym_qlm_buf = std::vector<std::vector<std::complex<double>>>(m_n_symmetries);
    for (auto& sym_qlm : sym_qlm_buf) {
        sym_qlm.reserve(qlm_eval.getNlm());
    }
    while (!brute_opt.terminate()) {
        const auto rotation = brute_opt.next_point();
        const auto particle_op = compute_pgop(rotation,
                                              position_begin,
                                              position_end,
                                              rotated_dist,
                                              sym_qlm_buf,
                                              qlm_eval);
        brute_opt.record_objective(score(particle_op));
    }
    const auto initial_simplex = getInitialSimplex(brute_opt.get_optimum().first);
    auto simplex_opt = NelderMead(NelderMeadParams(1.0, 2.0, 0.5, 0.5),
                                  initial_simplex,
                                  opt_min_bounds,
                                  opt_max_bounds,
                                  150,
                                  1e-3,
                                  1e-4);
    while (!simplex_opt.terminate()) {
        const auto rotation = simplex_opt.next_point();
        const auto particle_op = compute_pgop(rotation,
                                              position_begin,
                                              position_end,
                                              rotated_dist,
                                              sym_qlm_buf,
                                              qlm_eval);
        simplex_opt.record_objective(score(particle_op));
    }
    return compute_pgop(simplex_opt.get_optimum().first,
                        position_begin,
                        position_end,
                        rotated_dist,
                        sym_qlm_buf,
                        qlm_eval);
}

template<typename distribution_type>
std::vector<double>
PGOP<distribution_type>::compute_pgop(const std::vector<double>& rotation,
                                      const std::vector<Vec3>::const_iterator& position_begin,
                                      const std::vector<Vec3>::const_iterator& position_end,
                                      std::vector<Vec3>& rotated_positions,
                                      std::vector<std::vector<std::complex<double>>>& sym_qlm_buf,
                                      const QlmEval& qlm_eval) const
{
    rotate_euler(position_begin, position_end, rotated_positions.begin(), rotation);
    const auto bond_order = BondOrder<distribution_type>(getDistribution(), rotated_positions);
    const auto qlms = qlm_eval.eval<distribution_type>(bond_order);
    symmetrize_qlms(qlms, m_Dij, sym_qlm_buf, m_max_l);
    return covariance(qlms, sym_qlm_buf);
}

template<typename distribution_type>
std::vector<std::vector<double>> PGOP<distribution_type>::getDefaultRotations() const
{
    return std::vector<std::vector<double>> {
        std::initializer_list<double> {0, 0, 0},
        std::initializer_list<double> {-M_PI_2, -M_PI_4, -M_PI_2},
        std::initializer_list<double> {-M_PI_2, -M_PI_4, M_PI_2},
        std::initializer_list<double> {-M_PI_2, M_PI_4, -M_PI_2},
        std::initializer_list<double> {-M_PI_2, M_PI_4, M_PI_2},
        std::initializer_list<double> {M_PI_2, -M_PI_4, -M_PI_2},
        std::initializer_list<double> {M_PI_2, -M_PI_4, M_PI_2},
        std::initializer_list<double> {M_PI_2, M_PI_4, -M_PI_2},
        std::initializer_list<double> {M_PI_2, M_PI_4, M_PI_2}};
}

template<typename distribution_type>
std::vector<std::vector<double>>
PGOP<distribution_type>::getInitialSimplex(const std::vector<double>& center) const
{
    const double volume_fraction = 0.18333;
    const double scale = M_PI * std::pow(2.0, 1.0 / 6.0) * std::pow(3 * volume_fraction, 1.0 / 3.0);
    const double b = scale / std::sqrt(2);
    const double b_plus_z {b + center[2]}, z_minus_b {center[2] - b};
    return std::vector<std::vector<double>> {
        std::initializer_list<double> {center[0] + scale, center[1], z_minus_b},
        std::initializer_list<double> {center[0] - scale, center[1], z_minus_b},
        std::initializer_list<double> {center[0], center[1] + scale, b_plus_z},
        std::initializer_list<double> {center[0], center[1] - scale, b_plus_z}};
}

template<typename distribution_type>
distribution_type PGOP<distribution_type>::getDistribution() const
{
    return distribution_type(m_distribution_params);
}

template<typename distribution_type>
double PGOP<distribution_type>::score(const std::vector<double>& pgop) const
{
    return -m_p_norm->operator()(pgop);
}

template class PGOP<UniformDistribution>;
template class PGOP<FisherDistribution>;

template<typename distribution_type> void export_pgop_class(py::module& m, const std::string& name)
{
    py::class_<PGOP<distribution_type>>(m, name.c_str())
        .def(py::init([](unsigned int max_l,
                         const py::array_t<std::complex<double>> D_ij,
                         unsigned int p,
                         std::vector<double> pnorm_weights,
                         typename distribution_type::param_type distribution_params) {
            std::unique_ptr<WeightedPNormBase> p_norm(nullptr);
            if (p == 1) {
                p_norm.reset(new WeightedPNorm<1>(pnorm_weights));
            } else if (p == 2) {
                p_norm.reset(new WeightedPNorm<2>(pnorm_weights));
            } else if (p == 3) {
                p_norm.reset(new WeightedPNorm<3>(pnorm_weights));
            } else if (p == 4) {
                p_norm.reset(new WeightedPNorm<4>(pnorm_weights));
            } else {
                throw std::runtime_error("p cannot be greater than 4.");
            }
            return std::make_unique<PGOP<distribution_type>>(max_l,
                                                             D_ij,
                                                             std::move(p_norm),
                                                             distribution_params);
        }))
        .def("compute", &PGOP<distribution_type>::compute);
}

void export_pgop(py::module& m)
{
    export_pgop_class<UniformDistribution>(m, "PGOPUniform");
    export_pgop_class<FisherDistribution>(m, "PGOPFisher");
}
