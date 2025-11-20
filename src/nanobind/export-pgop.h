// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include "../PGOP.h"

#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
// #include <pybind11/numpy.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>

// namespace py = pybind11;
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
        [](nb::ndarray<double, nb::c_contig, nb::device::cpu> R_ij,
           size_t n_symmetries) { //,
                                  // std::shared_ptr<optimize::Optimizer> optimizer,
                                  // unsigned int mode,
                                  // bool compute_per_operator) {
            // Check dimensions if PGOP expects a specific rank, e.g., 2D
            // if (R_ij.ndim() != 2) throw std::runtime_error("R_ij must be 2-dimensional");

            return PGOP(R_ij.data(), // const double* R_ij_data
                        R_ij.size(), // const size_t R_ij_sizes
                        n_symmetries);
            // n_symmetries,
            // optimizer,
            // mode,
            // compute_per_operator);
        });
    // nb::arg("R_ij"),
    // nb::arg("n_symmetries"),
    // nb::arg("n_symmetries"));
    // nb::arg("optimizer"),
    // nb::arg("mode"),
    // nb::arg("compute_per_operator") = false);
}
} // End namespace spatula
