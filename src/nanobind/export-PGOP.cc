// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include "export-PGOP.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "../PGOP.h"
#include "../stores.h" // For PGOPStore

namespace nb = nanobind;
using namespace nb::literals;

namespace spatula {

void export_pgop(nb::module_& m) {
    nb::class_<PGOP>(m, "PGOP")
        .def(nb::init<const nb::list&,
                      std::shared_ptr<optimize::Optimizer>&,
                      const unsigned int,
                      bool>(),
             [](const nb::list& R_ij_list,
                std::shared_ptr<optimize::Optimizer>& optimizer,
                const unsigned int mode,
                bool compute_per_operator) {
            std::vector<std::vector<double>> R_ij;
            R_ij.reserve(R_ij_list.size());
            for (auto item : R_ij_list) {
                R_ij.emplace_back(nb::cast<std::vector<double>>(item));
            }
            return new PGOP(R_ij, optimizer, mode, compute_per_operator);
        },
             nb::arg("R_ij"),
             nb::arg("optimizer"),
             nb::arg("mode"),
             nb::arg("compute_per_operator"))
        .def("compute", [](const PGOP& pgop_instance,
                           const nb::ndarray<double, nb::ndim<2>, nb::c_contig> distances,
                           const nb::ndarray<double, nb::ndim<1>, nb::c_contig> weights,
                           const nb::ndarray<int, nb::ndim<1>, nb::c_contig> num_neighbors,
                           const nb::ndarray<double, nb::ndim<1>, nb::c_contig> sigmas) {
            if (distances.shape(0) != weights.shape(0) ||
                distances.shape(0) != num_neighbors.shape(0) ||
                distances.shape(0) != sigmas.shape(0)) {
                throw std::invalid_argument("Shape mismatch between distances, weights, num_neighbors, and sigmas");
            }
            return pgop_instance.compute(distances.shape(0),
                                         distances.data(),
                                         weights.data(),
                                         num_neighbors.data(),
                                         sigmas.data());
        },
             nb::arg("distances"),
             nb::arg("weights"),
             nb::arg("num_neighbors"),
             nb::arg("sigmas"));
}

} // namespace spatula
