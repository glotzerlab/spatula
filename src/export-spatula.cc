// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include "BOOSOP.h"
#include "BondOrder.h"
#include "PGOP.h"
#include "data/Quaternion.h"
#include "optimize/Optimize.h"
#include "pybind11/complex.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "util/QlmEval.h"
#include "util/Util.h"

namespace py = pybind11;

namespace spatula {

template<typename distribution_type>
void export_BOOSOP_class(py::module& m, const std::string& name)
{
    py::class_<BOOSOP<distribution_type>>(m, name.c_str())
        .def(py::init([](py::array_t<std::complex<double>> D_ij_array,
                         std::shared_ptr<optimize::Optimizer>& optimizer,
                         typename distribution_type::param_type distribution_params) {
            py::buffer_info buf_D_ij = D_ij_array.request();
            if (buf_D_ij.ndim != 2) {
                throw std::runtime_error("D_ij must be a 2D array");
            }
            std::vector<std::vector<std::complex<double>>> D_ij;
            D_ij.reserve(buf_D_ij.shape[0]);
            const std::complex<double>* D_ij_ptr = static_cast<std::complex<double>*>(buf_D_ij.ptr);
            for (size_t i = 0; i < buf_D_ij.shape[0]; ++i) {
                D_ij.emplace_back(
                    std::vector<std::complex<double>>(D_ij_ptr + i * buf_D_ij.shape[1],
                                                      D_ij_ptr + (i + 1) * buf_D_ij.shape[1]));
            }
            return new BOOSOP<distribution_type>(D_ij, optimizer, distribution_params);
        }), py::arg("D_ij"), py::arg("optimizer"), py::arg("distribution_params"))
        .def("compute", [](const BOOSOP<distribution_type>& self,
                           py::array_t<double> distances,
                           py::array_t<double> weights,
                           py::array_t<int> num_neighbors,
                           const unsigned int m,
                           py::array_t<std::complex<double>> ylms,
                           py::array_t<double> quad_positions,
                           py::array_t<double> quad_weights) {
            py::buffer_info buf_distances = distances.request();
            py::buffer_info buf_weights = weights.request();
            py::buffer_info buf_num_neighbors = num_neighbors.request();
            py::buffer_info buf_ylms = ylms.request();
            py::buffer_info buf_quad_positions = quad_positions.request();
            py::buffer_info buf_quad_weights = quad_weights.request();

            if (buf_distances.ndim != 1 || buf_weights.ndim != 1 || buf_num_neighbors.ndim != 1 || buf_ylms.ndim != 2 || buf_quad_positions.ndim != 1 || buf_quad_weights.ndim != 1) {
                throw std::runtime_error("Number of dimensions must be correct for BOOSOP::compute");
            }

            size_t N_points = buf_num_neighbors.size;

            const double* distances_ptr = static_cast<double*>(buf_distances.ptr);
            const double* weights_ptr = static_cast<double*>(buf_weights.ptr);
            const int* num_neighbors_ptr = static_cast<int*>(buf_num_neighbors.ptr);
            const std::complex<double>* ylms_ptr = static_cast<std::complex<double>*>(buf_ylms.ptr);
            const double* quad_positions_ptr = static_cast<double*>(buf_quad_positions.ptr);
            const double* quad_weights_ptr = static_cast<double*>(buf_quad_weights.ptr);

            auto result = self.compute(N_points,
                                       distances_ptr,
                                       weights_ptr,
                                       num_neighbors_ptr,
                                       m,
                                       ylms_ptr,
                                       quad_positions_ptr,
                                       quad_weights_ptr);

            std::vector<size_t> op_shape = {N_points, self.get_n_symmetries()};
            py::array_t<double> op_array(op_shape, result.first.data());

            std::vector<size_t> rotations_shape = {N_points, self.get_n_symmetries(), 4};
            py::array_t<double> rotations_array(rotations_shape, result.second.data());

            return py::make_tuple(op_array, rotations_array);
        }, py::arg("distances"), py::arg("weights"), py::arg("num_neighbors"), py::arg("m"), py::arg("ylms"), py::arg("quad_positions"), py::arg("quad_weights"))
        .def("refine", [](const BOOSOP<distribution_type>& self,
                           py::array_t<double> distances,
                           py::array_t<double> rotations,
                           py::array_t<double> weights,
                           py::array_t<int> num_neighbors,
                           const unsigned int m,
                           py::array_t<std::complex<double>> ylms,
                           py::array_t<double> quad_positions,
                           py::array_t<double> quad_weights) {
            py::buffer_info buf_distances = distances.request();
            py::buffer_info buf_rotations = rotations.request();
            py::buffer_info buf_weights = weights.request();
            py::buffer_info buf_num_neighbors = num_neighbors.request();
            py::buffer_info buf_ylms = ylms.request();
            py::buffer_info buf_quad_positions = quad_positions.request();
            py::buffer_info buf_quad_weights = quad_weights.request();

            if (buf_distances.ndim != 1 || buf_rotations.ndim != 3 || buf_weights.ndim != 1 || buf_num_neighbors.ndim != 1 || buf_ylms.ndim != 2 || buf_quad_positions.ndim != 1 || buf_quad_weights.ndim != 1) {
                throw std::runtime_error("Number of dimensions must be correct for BOOSOP::refine");
            }

            size_t N_points = buf_num_neighbors.size;

            const double* distances_ptr = static_cast<double*>(buf_distances.ptr);
            const double* rotations_ptr = static_cast<double*>(buf_rotations.ptr);
            const double* weights_ptr = static_cast<double*>(buf_weights.ptr);
            const int* num_neighbors_ptr = static_cast<int*>(buf_num_neighbors.ptr);
            const std::complex<double>* ylms_ptr = static_cast<std::complex<double>*>(buf_ylms.ptr);
            const double* quad_positions_ptr = static_cast<double*>(buf_quad_positions.ptr);
            const double* quad_weights_ptr = static_cast<double*>(buf_quad_weights.ptr);

            auto result = self.refine(N_points,
                                      distances_ptr,
                                      rotations_ptr,
                                      weights_ptr,
                                      num_neighbors_ptr,
                                      m,
                                      ylms_ptr,
                                      quad_positions_ptr,
                                      quad_weights_ptr);

            std::vector<size_t> op_shape = {N_points, self.get_n_symmetries()};
            py::array_t<double> op_array(op_shape, result.data());

            return op_array;
        }, py::arg("distances"), py::arg("rotations"), py::arg("weights"), py::arg("num_neighbors"), py::arg("m"), py::arg("ylms"), py::arg("quad_positions"), py::arg("quad_weights"));
}

void export_BOOSOP(py::module& m)
{
    export_BOOSOP_class<UniformDistribution>(m, "BOOSOPUniform");
    export_BOOSOP_class<FisherDistribution>(m, "BOOSOPFisher");
}

void export_PGOP(py::module& m)
{
    py::class_<PGOP>(m, "PGOP")
        .def(py::init([](const py::list& R_ij_list,
                         std::shared_ptr<optimize::Optimizer>& optimizer,
                         const unsigned int mode,
                         bool compute_per_operator) {
                 std::vector<std::vector<double>> R_ij;
                 for (const auto& item : R_ij_list) {
                     R_ij.push_back(py::cast<std::vector<double>>(item));
                 }
                 return new PGOP(R_ij, optimizer, mode, compute_per_operator);
             }),
             py::arg("R_ij"),
             py::arg("optimizer"),
             py::arg("mode"),
             py::arg("compute_per_operator"))
        .def(
            "compute",
            [](const PGOP& self,
               py::array_t<double> distances,
               py::array_t<double> weights,
               py::array_t<int> num_neighbors,
               py::array_t<double> sigmas) {
                py::buffer_info buf_distances = distances.request();
                py::buffer_info buf_weights = weights.request();
                py::buffer_info buf_num_neighbors = num_neighbors.request();
                py::buffer_info buf_sigmas = sigmas.request();

                if (buf_distances.ndim != 1 || buf_weights.ndim != 1 || buf_num_neighbors.ndim != 1
                    || buf_sigmas.ndim != 1) {
                    throw std::runtime_error("Number of dimensions must be one");
                }

                size_t N_points = buf_num_neighbors.size;

                const double* distances_ptr = static_cast<double*>(buf_distances.ptr);
                const double* weights_ptr = static_cast<double*>(buf_weights.ptr);
                const int* num_neighbors_ptr = static_cast<int*>(buf_num_neighbors.ptr);
                const double* sigmas_ptr = static_cast<double*>(buf_sigmas.ptr);

                auto result = self.compute(N_points,
                                           distances_ptr,
                                           weights_ptr,
                                           num_neighbors_ptr,
                                           sigmas_ptr);

                // Convert std::vector<double> to py::array_t<double>
                std::vector<size_t> op_shape = {N_points, self.get_n_symmetries()};
                py::array_t<double> op_array(op_shape, result.first.data());

                std::vector<size_t> rotations_shape = {N_points, self.get_n_symmetries(), 4};
                py::array_t<double> rotations_array(rotations_shape, result.second.data());

                return py::make_tuple(op_array, rotations_array);
            },
            py::arg("distances"),
            py::arg("weights"),
            py::arg("num_neighbors"),
            py::arg("sigmas"));
}

} // namespace spatula
