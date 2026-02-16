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
               const nb::ndarray<float, nb::shape<-1, 3>>& distances,
               const nb::ndarray<float, nb::shape<-1>>& weights,
               const nb::ndarray<int, nb::shape<-1>>& num_neighbors,
               const nb::ndarray<float, nb::shape<-1>>& sigmas

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
                const size_t num_op_values = op_values.size();
                double* op_data = new double[num_op_values];
                std::move(op_values.begin(), op_values.end(), op_data);

                nb::capsule op_owner(op_data,
                                     [](void* p) noexcept { delete[] static_cast<double*>(p); });

                nb::ndarray<nb::numpy, double, nb::shape<-1, -1>> py_op_values(
                    op_data,
                    {num_query_points, num_symmetries},
                    op_owner);

                // Convert rotation_values to ndarray
                double* rotation_data = new double[rotation_values.size() * 4];
                for (size_t i = 0; i < rotation_values.size(); ++i) {
                    rotation_data[i * 4 + 0] = rotation_values[i].w;
                    rotation_data[i * 4 + 1] = rotation_values[i].x;
                    rotation_data[i * 4 + 2] = rotation_values[i].y;
                    rotation_data[i * 4 + 3] = rotation_values[i].z;
                }

                // Create capsule with deleter to own the memory
                nb::capsule rotation_owner(rotation_data, [](void* p) noexcept {
                    delete[] static_cast<double*>(p);
                });

                nb::ndarray<nb::numpy, double, nb::shape<-1, -1, 4>> py_rotations(
                    rotation_data,
                    {num_query_points, num_symmetries, 4},
                    rotation_owner);

                return nb::make_tuple(py_op_values, py_rotations);
            },
            nb::arg("distances"),
            nb::arg("weights"),
            nb::arg("neighbor_counts"),
            nb::arg("sigmas"));
}
} // End namespace spatula
