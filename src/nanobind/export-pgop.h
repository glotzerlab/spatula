// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include "../PGOP.h"

#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
namespace nb = nanobind;
using namespace nb::literals;

namespace spatula {

void export_spatula(nb::module_& m)
{
    nb::class_<PGOP>(m, "PGOP")
        .def(
            "__init__",
            [](PGOP* self,
               const std::vector<nb::ndarray<double, nb::ndim<1>>>& R_ij_list,
               std::shared_ptr<optimize::Optimizer>& optimizer,
               unsigned int m_mode,
               bool compute_per_operator) {
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

                new (self) PGOP(ptrs,
                                R_ij_list.size(),
                                optimizer,
                                group_sizes,
                                m_mode,
                                compute_per_operator);
            },
            // Keep argument 2 (symops) alive as long as arg 1 (self) is alive
            nb::keep_alive<1, 2>(),
            nb::arg("R_ij"),
            nb::arg("optimizer"),
            nb::arg("m_mode"),
            nb::arg("compute_per_operator"))
        .def(
            "compute",
            [](PGOP* self,
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

                // Convert std::vector<Quaternion> to ndarray with shape (Nq, Ns, 4)
                const size_t num_query_points = num_neighbors.size();
                const size_t num_symmetries = rotation_values.size() / num_query_points;

                // Convert op_values to ndarray with shape (Nq, Ns)
                nb::ndarray<nb::numpy, double, nb::shape<-1, -1>> py_op_values
                    = nb::ndarray<double, nb::shape<-1, -1>>(op_values.data(),
                                                             {num_query_points, num_symmetries})
                          .cast();

                // Convert rotation_values to ndarray
                nb::ndarray<nb::numpy, double, nb::shape<-1, -1, 4>> py_rotations(
                    nullptr,
                    {num_query_points, num_symmetries, 4});

                double* ptr = py_rotations.data();
                for (const auto& val : rotation_values) {
                    ptr[0] = val.w;
                    ptr[1] = val.x;
                    ptr[2] = val.y;
                    ptr[3] = val.z;
                    ptr += 4;
                }

                return nb::make_tuple(py_op_values, py_rotations);
            },
            nb::arg("distances"),
            nb::arg("weights"),
            nb::arg("neighbor_counts"),
            nb::arg("sigmas"));
}
} // End namespace spatula
