// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include "../PGOP.h"

#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;
using namespace nb::literals;

namespace spatula {

// We reinterpret_cast an array of quaternions to pass to numpy without copying.
// These are a few sanity checks to make sure this is safe.
static_assert(std::is_standard_layout_v<spatula::data::Quaternion>,
              "Quaternion is not standard layout!");
static_assert(sizeof(spatula::data::Quaternion) == 4 * sizeof(double),
              "Quaternion struct has unexpected padding!");

using QuaternionArrayXd = nb::ndarray<double, nb::numpy, nb::shape<-1, 4>>;
using NBArrayXXd = nb::ndarray<double, nb::numpy, nb::shape<-1, -1>>;

void export_spatula(nb::module_& m)
{
    nb::class_<PGOP>(m, "PGOP")
        .def(
            "__init__",
            [](PGOP* self,
               const std::vector<nb::ndarray<double, nb::ndim<1>>>& R_ij_list,
               std::shared_ptr<optimize::Optimizer>& optimizer) {
                // Save a vector of pointers to each group's data
                const std::vector<const double*> ptrs = [&]() {
                    std::vector<const double*> v;
                    v.reserve(R_ij_list.size());
                    for (const auto& arr : R_ij_list)
                        v.push_back(arr.data());
                    return v;
                }();

                // Initialize our vector of group sizes (9 * G)
                const std::vector<size_t> group_sizes = [&]() {
                    std::vector<size_t> v;
                    v.reserve(R_ij_list.size());
                    for (const auto& arr : R_ij_list)
                        v.push_back(arr.size());
                    return v;
                }();

                new (self) PGOP(ptrs, R_ij_list.size(), optimizer, group_sizes);
            },
            // Keep argument 2 (symops) alive as long as arg 1 (self) is alive
            nb::keep_alive<1, 2>(),
            nb::arg("R_ij"),
            // nb::arg("n_symmetries"),
            // nb::arg("n_symmetries"));
            // nb::arg("optimizer"),
            nb::arg("optimizer"))
        .def(
            "compute",
            []( /// TODO
                PGOP* self,
                const nb::ndarray<double, nb::shape<-1, 3>>& distances,
                const nb::ndarray<double, nb::shape<-1>>& weights,
                const nb::ndarray<int, nb::shape<-1>>& num_neighbors,
                const nb::ndarray<double, nb::shape<-1>>& sigmas

            ) {
                auto results_tuple = self->compute(distances.data(),
                                                   weights.data(),
                                                   num_neighbors.data(),
                                                   sigmas.data(),
                                                   num_neighbors.size());
                // Extract the std::vectors from the tuple
                const auto& [op_values, rotation_values] = results_tuple;
                return nb::make_tuple(op_values, rotation_values);
            },
            nb::arg("distances"),
            nb::arg("weights"),
            nb::arg("neighbor_counts"),
            nb::arg("sigmas"));
    // nb::arg("mode"),
    // nb::arg("compute_per_operator") = false);
}
} // End namespace spatula
