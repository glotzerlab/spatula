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

/**
 * @brief Wrapper function to call PGOP::compute and convert its std::tuple output to nb::tuple.
 * This function is used for exposing the compute method to Python.
 */
// nb::tuple wrap_pgop_compute(const PGOP& pgop_instance,
void wrap_pgop_compute(const PGOP& pgop_instance,
                       const nb::ndarray<double, nb::shape<-1, 3>>& distances,
                       const nb::ndarray<double, nb::shape<-1>>& weights,
                       const nb::ndarray<int, nb::shape<-1>>& num_neighbors,
                       const nb::ndarray<double, nb::shape<-1>>& sigmas)
{
    // Call the C++ compute method
    auto results_tuple = pgop_instance.compute(distances.data(),
                                               weights.data(),
                                               num_neighbors.data(),
                                               sigmas.data(),
                                               num_neighbors.size());

    // Extract the std::vectors from the tuple
    [[maybe_unused]] const auto& op_values = std::get<0>(results_tuple);
    [[maybe_unused]] const auto& rotation_values = std::get<1>(results_tuple);

    // Determine sizes for py::array_t
    // N_particles can be inferred from num_neighbors.size()
    // const size_t N_particles = num_neighbors.size();
    // ops_per_particle can be inferred from op_values.size() / N_particles
    // const size_t ops_per_particle = op_values.empty() ? 0 : op_values.size() / N_particles;

    // Convert std::vector<double> to py::array_t<double>
    // py::array_t<double> result_ops_py;
    // result_ops_py.resize(std::vector<ssize_t> {static_cast<ssize_t>(N_particles),
    // static_cast<ssize_t>(ops_per_particle)});
    // std::copy(op_values.begin(), op_values.end(), result_ops_py.mutable_data());

    // Convert std::vector<data::Quaternion> to py::array_t<double> with shape (N, ops, 4)
    std::vector<double> flat_rotation_components;
    flat_rotation_components.reserve(rotation_values.size() * 4);
    for (const auto& q : rotation_values) {
        flat_rotation_components.push_back(q.w);
        flat_rotation_components.push_back(q.x);
        flat_rotation_components.push_back(q.y);
        flat_rotation_components.push_back(q.z);
    }

    // TODO: WIP: how to port with 0 copy?
    // py::array_t<double> result_rots_py;
    // result_rots_py.resize(std::vector<ssize_t> {static_cast<ssize_t>(N_particles),
    // static_cast<ssize_t>(ops_per_particle), 4
    // });
    // std::copy(flat_rotation_components.begin(),
    // flat_rotation_components.end(),
    // result_rots_py.mutable_data());

    // return nb::make_tuple(result_ops_py, result_rots_py);
} // namespace spatula

void export_spatula(nb::module_& m)
{
    nb::class_<PGOP>(m, "PGOP").def(
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
            const std::vector<size_t> lengths = [&]() {
                std::vector<size_t> v;
                v.reserve(R_ij_list.size());
                for (const auto& arr : R_ij_list)
                    v.push_back(arr.size());
                return v;
            }();

            new (self) PGOP(ptrs, R_ij_list.size(), optimizer, lengths);
        },
        // Keep argument 2 (symops) alive as long as arg 1 (self) is alive
        nb::keep_alive<1, 2>());
    // nb::arg("R_ij"),
    // nb::arg("n_symmetries"),
    // nb::arg("n_symmetries"));
    // nb::arg("optimizer"),
    // nb::arg("mode"),
    // nb::arg("compute_per_operator") = false);
}
} // End namespace spatula
