// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <memory>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "BOOSOP.h"
#include "BondOrder.h"
#include "optimize/Optimize.h"

#include "export-boosop.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace spatula {

namespace {

template<typename distribution_type>
void export_BOOSOP_class(nb::module_& m, const std::string& name)
{
    nb::class_<BOOSOP<distribution_type>>(m, name.c_str())
        .def(
            "__init__",
            [](BOOSOP<distribution_type>* self,
               const nb::ndarray<std::complex<double>, nb::ndim<2>, nb::c_contig> D_ij_py,
               std::shared_ptr<optimize::Optimizer>& optimizer,
               typename distribution_type::param_type distribution_params) {
                const size_t n_symmetries = D_ij_py.shape(0);
                const size_t n_mlms = D_ij_py.shape(1);
                std::vector<std::vector<std::complex<double>>> D_ij;
                D_ij.reserve(n_symmetries);

                for (size_t i {0}; i < n_symmetries; ++i) {
                    D_ij.emplace_back(
                        std::vector<std::complex<double>>(D_ij_py.data() + i * n_mlms,
                                                          D_ij_py.data() + (i + 1) * n_mlms));
                }

                new (self) BOOSOP<distribution_type>(D_ij, optimizer, distribution_params);
            },
            nb::keep_alive<1, 2>(),
            "D_ij"_a,
            "optimizer"_a,
            "distribution_params"_a)
        .def("compute",
             [](const BOOSOP<distribution_type>& self,
                const nb::ndarray<double, nb::ndim<1>, nb::c_contig> distances,
                const nb::ndarray<double, nb::ndim<1>, nb::c_contig> weights,
                const nb::ndarray<int, nb::ndim<1>, nb::c_contig> num_neighbors,
                const unsigned int m,
                const nb::ndarray<std::complex<double>, nb::ndim<2>, nb::c_contig> ylms,
                const nb::ndarray<double, nb::ndim<1>, nb::c_contig> quad_positions,
                const nb::ndarray<double, nb::ndim<1>, nb::c_contig> quad_weights) {
                 auto result_tuple = self.compute(distances.data(),
                                                  weights.data(),
                                                  num_neighbors.data(),
                                                  num_neighbors.shape(0),
                                                  m,
                                                  ylms.data(),
                                                  ylms.shape(0),
                                                  quad_positions.data(),
                                                  quad_positions.shape(0),
                                                  quad_weights.data());

                 const auto& op_values = std::get<0>(result_tuple);
                 const auto& rotation_values = std::get<1>(result_tuple);

                 nb::list py_rotations;
                 for (const auto& q : rotation_values) {
                     py_rotations.append(nb::make_tuple(q.w, q.x, q.y, q.z));
                 }

                 return nb::make_tuple(nb::cast(op_values), py_rotations);
             })
        .def("refine",
             [](const BOOSOP<distribution_type>& self,
                const nb::ndarray<double, nb::ndim<1>, nb::c_contig> distances,
                const nb::ndarray<double, nb::ndim<1>, nb::c_contig> rotations,
                const nb::ndarray<double, nb::ndim<1>, nb::c_contig> weights,
                const nb::ndarray<int, nb::ndim<1>, nb::c_contig> num_neighbors,
                const unsigned int m,
                const nb::ndarray<std::complex<double>, nb::ndim<2>, nb::c_contig> ylms,
                const nb::ndarray<double, nb::ndim<1>, nb::c_contig> quad_positions,
                const nb::ndarray<double, nb::ndim<1>, nb::c_contig> quad_weights) {
                 auto op_values_vec = self.refine(distances.data(),
                                                  rotations.data(),
                                                  weights.data(),
                                                  num_neighbors.data(),
                                                  num_neighbors.shape(0),
                                                  m,
                                                  ylms.data(),
                                                  ylms.shape(0),
                                                  quad_positions.data(),
                                                  quad_positions.shape(0),
                                                  quad_weights.data());

                 return nb::cast(op_values_vec);
             });
}
} // namespace

void export_BOOSOP(nb::module_& m)
{
    export_BOOSOP_class<UniformDistribution>(m, "BOOSOPUniform");
    export_BOOSOP_class<FisherDistribution>(m, "BOOSOPFisher");
}

} // namespace spatula
