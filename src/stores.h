// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <tuple>
#include <vector>
#include <limits>

#include "data/Quaternion.h"

namespace py = pybind11;

namespace spatula {

struct BOOSOPStore {
    BOOSOPStore(size_t N_particles, size_t N_symmetries)
        : N_syms(N_symmetries), op(std::vector<size_t> {N_particles, N_symmetries}),
          rotations(std::vector<size_t> {N_particles, N_symmetries, 4}),
          u_op(op.mutable_unchecked<2>()), u_rotations(rotations.mutable_unchecked<3>())
    {
    }

    size_t N_syms;
    py::array_t<double> op;
    py::array_t<double> rotations;

    void addOp(size_t i, const std::tuple<std::vector<double>, std::vector<data::Quaternion>>& op_)
    {
        const auto& values = std::get<0>(op_);
        const auto& rots = std::get<1>(op_);
        for (size_t j {0}; j < N_syms; ++j) {
            u_op(i, j) = values[j];
            u_rotations(i, j, 0) = rots[j].w;
            u_rotations(i, j, 1) = rots[j].x;
            u_rotations(i, j, 2) = rots[j].y;
            u_rotations(i, j, 3) = rots[j].z;
        }
    }

    void addNull(size_t i)
    {
        for (size_t j {0}; j < N_syms; ++j) {
            u_op(i, j) = 0;
            u_rotations(i, j, 0) = 1;
            u_rotations(i, j, 1) = 0;
            u_rotations(i, j, 2) = 0;
            u_rotations(i, j, 3) = 0;
        }
    }

    py::tuple getArrays()
    {
        return py::make_tuple(op, rotations);
    }

    private:
    py::detail::unchecked_mutable_reference<double, 2> u_op;
    py::detail::unchecked_mutable_reference<double, 3> u_rotations;
};

struct PGOPStore {
    PGOPStore(size_t N_particles, size_t N_symmetries)
        : N_syms(N_symmetries), op(std::vector<size_t> {N_particles, N_symmetries}),
          rotations(std::vector<size_t> {N_particles, N_symmetries, 4}),
          u_op(op.mutable_unchecked<2>()), u_rotations(rotations.mutable_unchecked<3>())
    {
    }

    size_t N_syms;
    py::array_t<double> op;
    py::array_t<double> rotations;

    void addOp(size_t i, const std::tuple<std::vector<double>, std::vector<data::Quaternion>>& op_)
    {
        const auto& values = std::get<0>(op_);
        const auto& rots = std::get<1>(op_);
        for (size_t j {0}; j < N_syms; ++j) {
            u_op(i, j) = values[j];
            u_rotations(i, j, 0) = rots[j].w;
            u_rotations(i, j, 1) = rots[j].x;
            u_rotations(i, j, 2) = rots[j].y;
            u_rotations(i, j, 3) = rots[j].z;
        }
    }

    void addNull(size_t i)
    {
        for (size_t j {0}; j < N_syms; ++j) {
            u_op(i, j) = std::numeric_limits<double>::quiet_NaN(); // Set NaN
            u_rotations(i, j, 0) = 1;
            u_rotations(i, j, 1) = 0;
            u_rotations(i, j, 2) = 0;
            u_rotations(i, j, 3) = 0;
        }
    }

    py::tuple getArrays()
    {
        return py::make_tuple(op, rotations);
    }

    private:
    py::detail::unchecked_mutable_reference<double, 2> u_op;
    py::detail::unchecked_mutable_reference<double, 3> u_rotations;
};

} // namespace spatula
