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
py::tuple BOOSOP<distribution_type>::compute(const py::array_t<double> distances,
                                             const py::array_t<double> weights,
                                             const py::array_t<int> num_neighbors,
                                             const unsigned int m,
                                             const py::array_t<std::complex<double>> ylms,
                                             const py::array_t<double> quad_positions,
                                             const py::array_t<double> quad_weights) const
{
    const auto qlm_eval = util::QlmEval(m,
                                        quad_positions.data(),
                                        quad_weights.data(),
                                        ylms.data(),
                                        quad_positions.shape(0),
                                        ylms.shape(0));
    const auto neighborhoods = NeighborhoodBOOs(num_neighbors.size(),
                                                num_neighbors.data(0),
                                                weights.data(0),
                                                distances.data(0));
    const size_t N_particles = num_neighbors.size();

    py::array_t<double> op(std::vector<size_t> {N_particles, m_n_symmetries});
    py::array_t<double> rotations(std::vector<size_t> {N_particles, m_n_symmetries, 4});
    auto u_op = op.mutable_unchecked<2>();
    auto u_rotations = rotations.mutable_unchecked<3>();

    const auto loop_func = [&u_op, &u_rotations, &neighborhoods, &qlm_eval, this](
        const size_t start, const size_t stop) {
        auto qlm_buf = util::QlmBuf(qlm_eval.getNlm());
        for (size_t i = start; i < stop; ++i)
        {
            if (neighborhoods.getNeighborCount(i) == 0)
            {
                for (size_t j {0}; j < m_n_symmetries; ++j)
                {
                    u_op(i, j) = 0;
                    u_rotations(i, j, 0) = 1;
                    u_rotations(i, j, 1) = 0;
                    u_rotations(i, j, 2) = 0;
                    u_rotations(i, j, 3) = 0;
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
                u_op(i, j) = values[j];
                u_rotations(i, j, 0) = rots[j].w;
                u_rotations(i, j, 1) = rots[j].x;
                u_rotations(i, j, 2) = rots[j].y;
                u_rotations(i, j, 3) = rots[j].z;
            }
        }
    };
    execute_func(loop_func, N_particles);
    return py::make_tuple(op, rotations);
}

template<typename distribution_type>
py::array_t<double> BOOSOP<distribution_type>::refine(const py::array_t<double> distances,
                                                      const py::array_t<double> rotations,
                                                      const py::array_t<double> weights,
                                                      const py::array_t<int> num_neighbors,
                                                      const unsigned int m,
                                                      const py::array_t<std::complex<double>> ylms,
                                                      const py::array_t<double> quad_positions,
                                                      const py::array_t<double> quad_weights) const
{
    const auto qlm_eval = util::QlmEval(m,
                                        quad_positions.data(),
                                        quad_weights.data(),
                                        ylms.data(),
                                        quad_positions.shape(0),
                                        ylms.shape(0));
    const auto neighborhoods = NeighborhoodBOOs(num_neighbors.size(),
                                                num_neighbors.data(0),
                                                weights.data(0),
                                                distances.data(0));
    const size_t N_particles = num_neighbors.size();
    py::array_t<double> op_store(std::vector<size_t> {N_particles, m_n_symmetries});
    auto u_op_store = op_store.mutable_unchecked<2>();
    auto u_rotations = rotations.unchecked<3>();
    const auto loop_func
        = [&u_op_store, &u_rotations, &neighborhoods, &qlm_eval, this](const size_t start,
                                                                       const size_t stop) {
              auto qlm_buf = util::QlmBuf(qlm_eval.getNlm());
              for (size_t i = start; i < stop; ++i) {
                  if (neighborhoods.getNeighborCount(i) == 0) {
                      for (size_t j {0}; j < m_n_symmetries; ++j) {
                          u_op_store(i, j) = 0;
                      }
                      continue;
                  }
                  auto neighborhood = neighborhoods.getNeighborhoodBOO(i);
                  for (size_t j {0}; j < m_n_symmetries; ++j) {
                      const auto rot = data::Quaternion(u_rotations(i, j, 0),
                                                        u_rotations(i, j, 1),
                                                        u_rotations(i, j, 2),
                                                        u_rotations(i, j, 3))
                                           .to_axis_angle_3D();
                      neighborhood.rotate(rot);
                      u_op_store(i, j)
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

// Explicit template instantiations for BondOrder
template class BondOrder<UniformDistribution>;
template class BondOrder<FisherDistribution>;

template<typename distribution_type>
void export_BOOSOP_class(py::module& m, const std::string& name)
{
    py::class_<BOOSOP<distribution_type>>(m, name.c_str())
        .def(py::init<const py::array_t<std::complex<double>>,
                      std::shared_ptr<optimize::Optimizer>&,
                      typename distribution_type::param_type>())
        .def("compute", &BOOSOP<distribution_type>::compute)
        .def("refine", &BOOSOP<distribution_type>::refine);
}

void export_BOOSOP(py::module& m)
{
    export_BOOSOP_class<UniformDistribution>(m, "BOOSOPUniform");
    export_BOOSOP_class<FisherDistribution>(m, "BOOSOPFisher");
}
} // namespace spatula
