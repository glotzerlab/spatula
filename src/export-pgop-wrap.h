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
    auto results_tuple = pgop_instance.compute(distances, weights, num_neighbors, sigmas);

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
    result_ops_py.resize(std::vector<ssize_t>{static_cast<ssize_t>(N_particles), static_cast<ssize_t>(ops_per_particle)});
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
    result_rots_py.resize(std::vector<ssize_t>{static_cast<ssize_t>(N_particles), static_cast<ssize_t>(ops_per_particle), 4});
    std::copy(flat_rotation_components.begin(), flat_rotation_components.end(), result_rots_py.mutable_data());

    return py::make_tuple(result_ops_py, result_rots_py);
}

} // End namespace spatula
