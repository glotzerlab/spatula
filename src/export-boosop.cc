// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <memory>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "BOOSOP.h"
#include "BondOrder.h"
#include "optimize/Optimize.h"

#include "export-boosop.h"

namespace py = pybind11;

namespace spatula {

namespace {

template<typename distribution_type>
void export_BOOSOP_class(py::module& m, const std::string& name)
{
    py::class_<BOOSOP<distribution_type>>(m, name.c_str())
        .def(py::init([](const py::array_t<std::complex<double>> D_ij_py,
                         std::shared_ptr<optimize::Optimizer>& optimizer,
                         typename distribution_type::param_type distribution_params) {
            const auto u_D_ij = D_ij_py.unchecked<2>();
            const size_t n_symmetries = D_ij_py.shape(0);
            const size_t n_mlms = D_ij_py.shape(1);
            std::vector<std::vector<std::complex<double>>> D_ij;
            D_ij.reserve(n_symmetries);

            for (size_t i {0}; i < n_symmetries; ++i)
            {
                D_ij.emplace_back(std::vector<std::complex<double>>(
                    u_D_ij.data(i, 0), u_D_ij.data(i, 0) + n_mlms));
            }

            return new BOOSOP<distribution_type>(D_ij, optimizer, distribution_params);
        }))
        .def("compute",
             [](const BOOSOP<distribution_type>& self,
                const py::array_t<double> distances,
                const py::array_t<double> weights,
                const py::array_t<int> num_neighbors,
                const unsigned int m,
                const py::array_t<std::complex<double>> ylms,
                const py::array_t<double> quad_positions,
                const py::array_t<double> quad_weights) {
                 auto result_tuple = self.compute(distances.data(0),
                                                  weights.data(0),
                                                  num_neighbors.data(0),
                                                  num_neighbors.size(),
                                                  m,
                                                  ylms.data(0),
                                                  ylms.shape(0),
                                                  quad_positions.data(0),
                                                  quad_positions.shape(0),
                                                  quad_weights.data(0));

                 const auto& op_values = std::get<0>(result_tuple);
                 const auto& rotation_values = std::get<1>(result_tuple);
                 const size_t N_particles = num_neighbors.size();
                 const size_t n_symmetries = op_values.size() / N_particles;

                 py::array_t<double> op_arr({N_particles, n_symmetries});
                 py::array_t<double> rot_arr({N_particles, n_symmetries, 4});

                 auto op_ptr = op_arr.mutable_unchecked<2>();
                 auto rot_ptr = rot_arr.mutable_unchecked<3>();

                 for (size_t i = 0; i < N_particles; ++i)
                 {
                     for (size_t j = 0; j < n_symmetries; ++j)
                     {
                         op_ptr(i, j) = op_values[i * n_symmetries + j];
                         const auto& rot = rotation_values[i * n_symmetries + j];
                         rot_ptr(i, j, 0) = rot.w;
                         rot_ptr(i, j, 1) = rot.x;
                         rot_ptr(i, j, 2) = rot.y;
                         rot_ptr(i, j, 3) = rot.z;
                     }
                 }

                 return py::make_tuple(op_arr, rot_arr);
             })
        .def("refine",
             [](const BOOSOP<distribution_type>& self,
                const py::array_t<double> distances,
                const py::array_t<double> rotations,
                const py::array_t<double> weights,
                const py::array_t<int> num_neighbors,
                const unsigned int m,
                const py::array_t<std::complex<double>> ylms,
                const py::array_t<double> quad_positions,
                const py::array_t<double> quad_weights) {
                 auto op_values_vec = self.refine(distances.data(0),
                                                    rotations.data(0),
                                                    weights.data(0),
                                                    num_neighbors.data(0),
                                                    num_neighbors.size(),
                                                    m,
                                                    ylms.data(0),
                                                    ylms.shape(0),
                                                    quad_positions.data(0),
                                                    quad_positions.shape(0),
                                                    quad_weights.data(0));

                 const size_t N_particles = num_neighbors.size();
                 const size_t n_symmetries = op_values_vec.size() / N_particles;
                 py::array_t<double> op_store({N_particles, n_symmetries});
                 auto u_op_store = op_store.mutable_unchecked<2>();
                 for (size_t i = 0; i < N_particles; ++i)
                 {
                     for (size_t j = 0; j < n_symmetries; ++j)
                     {
                         u_op_store(i, j) = op_values_vec[i * n_symmetries + j];
                     }
                 }
                 return op_store;
             });
}
} // namespace

void export_BOOSOP(py::module& m)
{
    export_BOOSOP_class<UniformDistribution>(m, "BOOSOPUniform");
    export_BOOSOP_class<FisherDistribution>(m, "BOOSOPFisher");
}

} // namespace spatula
