// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include "../PGOP.h"

#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
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
    const auto& op_values = std::get<0>(results_tuple);
    const auto& rotation_values = std::get<1>(results_tuple);

    // Determine sizes for py::array_t
    // N_particles can be inferred from num_neighbors.size()
    const size_t N_particles = num_neighbors.size();
    // ops_per_particle can be inferred from op_values.size() / N_particles
    const size_t ops_per_particle = op_values.empty() ? 0 : op_values.size() / N_particles;

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
    nb::class_<PGOP>(m, "PGOP").def(nb::init([](const nb::ndarray<>& R_ij,
                                                std::shared_ptr<optimize::Optimizer>& optimizer,
                                                const unsigned int mode,
                                                bool compute_per_operator) {
                                        std::vector<double> R_ij_data_vec;
                                        std::vector<size_t> R_ij_sizes_vec;
                                        size_t n_symmetries = R_ij.size(); // TODO: incorrect

                                        for (size_t i = 0; i < R_ij.shape(0); ++i) {
                                            for (size_t j = 0; j < R_ij.shape(1); ++j) {
                                                for (size_t k = 0; k < R_ij.shape(2);
                                                     ++k) { // 9 elements for
                                                    R_ij_data_vec.push_back(R_ij.view(i, j, k));
                                                }
                                            }
                                        }
                                        return std::make_unique<PGOP>(R_ij_data_vec.data(),
                                                                      R_ij_sizes_vec.data(),
                                                                      n_symmetries,
                                                                      optimizer,
                                                                      mode,
                                                                      compute_per_operator);
                                    }),
                                    "R_ij"_a,
                                    "optimizer"_a,
                                    "mode"_a,
                                    "compute_per_operator"_a);
    // .def("compute",
    //      &wrap_pgop_compute,
    //      py::arg("distances"),
    //      py::arg("weights"),
    //      py::arg("num_neighbors"),
    //      py::arg("sigmas"),
    //      "Compute PGOP values and rotations for a set of points.");
}

} // End namespace spatula
