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
        }), py::arg("R_ij"), py::arg("optimizer"), py::arg("mode"), py::arg("compute_per_operator"))
        .def("compute", [](const PGOP& self,
                           py::array_t<double> distances,
                           py::array_t<double> weights,
                           py::array_t<int> num_neighbors,
                           py::array_t<double> sigmas) {
            py::buffer_info buf_distances = distances.request();
            py::buffer_info buf_weights = weights.request();
            py::buffer_info buf_num_neighbors = num_neighbors.request();
            py::buffer_info buf_sigmas = sigmas.request();

            if (buf_distances.ndim != 1 || buf_weights.ndim != 1 || buf_num_neighbors.ndim != 1 || buf_sigmas.ndim != 1) {
                throw std::runtime_error("Number of dimensions must be one");
            }

            size_t N_points = buf_num_neighbors.size;

            const double* distances_ptr = static_cast<double*>(buf_distances.ptr);
            const double* weights_ptr = static_cast<double*>(buf_weights.ptr);
            const int* num_neighbors_ptr = static_cast<int*>(buf_num_neighbors.ptr);
            const double* sigmas_ptr = static_cast<double*>(buf_sigmas.ptr);

            auto result = self.compute(N_points, distances_ptr, weights_ptr, num_neighbors_ptr, sigmas_ptr);

            // Convert std::vector<double> to py::array_t<double>
            std::vector<size_t> op_shape = {N_points, self.get_n_symmetries()};
            py::array_t<double> op_array(op_shape, result.first.data());

            std::vector<size_t> rotations_shape = {N_points, self.get_n_symmetries(), 4};
            py::array_t<double> rotations_array(rotations_shape, result.second.data());

            return py::make_tuple(op_array, rotations_array);
        }, py::arg("distances"), py::arg("weights"), py::arg("num_neighbors"), py::arg("sigmas"));
}

} // namespace spatula
