// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include "PGOP.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace spatula {

/**
 * @brief Wrapper function to call PGOP::compute and convert its std::tuple output to py::tuple.
 * This function is used for exposing the compute method to Python.
 */
py::tuple wrap_pgop_compute(const PGOP& pgop_instance,
                            const py::array_t<double> distances,
                            const py::array_t<double> weights,
                            const py::array_t<int> num_neighbors,
                            const py::array_t<double> sigmas)
{
    // Call the C++ compute method
    auto results_tuple = pgop_instance.compute(distances.data(0),
                                               weights.data(0),
                                               num_neighbors.data(0),
                                               sigmas.data(0),
                                               num_neighbors.size());

    // Extract the std::vectors from the tuple
    const auto& op_values = std::get<0>(results_tuple);
    const auto& rotation_values = std::get<1>(results_tuple);

    // Determine sizes for py::array_t
    // N_particles can be inferred from num_neighbors.size()
    const size_t N_particles = num_neighbors.size();
    // ops_per_particle can be inferred from op_values.size() / N_particles
    const size_t ops_per_particle = op_values.empty() ? 0 : op_values.size() / N_particles;

    // Convert std::vector<double> to py::array_t<double>
    py::array_t<double> result_ops_py;
    result_ops_py.resize(std::vector<ssize_t> {static_cast<ssize_t>(N_particles),
                                               static_cast<ssize_t>(ops_per_particle)});
    std::copy(op_values.begin(), op_values.end(), result_ops_py.mutable_data());

    // Convert std::vector<data::Quaternion> to py::array_t<double> with shape (N, ops, 4)
    std::vector<double> flat_rotation_components;
    flat_rotation_components.reserve(rotation_values.size() * 4);
    for (const auto& q : rotation_values) {
        flat_rotation_components.push_back(q.w);
        flat_rotation_components.push_back(q.x);
        flat_rotation_components.push_back(q.y);
        flat_rotation_components.push_back(q.z);
    }

    py::array_t<double> result_rots_py;
    result_rots_py.resize(std::vector<ssize_t> {static_cast<ssize_t>(N_particles),
                                                static_cast<ssize_t>(ops_per_particle),
                                                4});
    std::copy(flat_rotation_components.begin(),
              flat_rotation_components.end(),
              result_rots_py.mutable_data());

    return py::make_tuple(result_ops_py, result_rots_py);
}

void export_spatula_class(py::module& m, const std::string& name)
{
    pybind11::class_<PGOP>(m, name.c_str())
        .def(pybind11::init([](const pybind11::list& R_ij,
                               std::shared_ptr<optimize::Optimizer>& optimizer,
                               const unsigned int mode,
                               bool compute_per_operator) {
                 std::vector<double> R_ij_data_vec;
                 std::vector<size_t> R_ij_sizes_vec;
                 size_t n_symmetries = R_ij.size();

                 for (size_t i = 0; i < n_symmetries; ++i) {
                     py::list inner_list = R_ij[i].cast<py::list>();
                     R_ij_sizes_vec.push_back(inner_list.size());
                     for (size_t j = 0; j < inner_list.size(); ++j) {
                         R_ij_data_vec.push_back(inner_list[j].cast<double>());
                     }
                 }
                 return std::make_unique<PGOP>(
                     R_ij_data_vec.data(), R_ij_sizes_vec.data(), n_symmetries, optimizer, mode,
                     compute_per_operator);
             }),
             py::arg("R_ij"),
             py::arg("optimizer"),
             py::arg("mode"),
             py::arg("compute_per_operator"),
             "Constructor for PGOP")
        .def("compute",
             &wrap_pgop_compute,
             py::arg("distances"),
             py::arg("weights"),
             py::arg("num_neighbors"),
             py::arg("sigmas"),
             "Compute PGOP values and rotations for a set of points.");
}

void export_spatula(pybind11::module& m)
{
    export_spatula_class(m, "PGOP");
}

} // End namespace spatula
