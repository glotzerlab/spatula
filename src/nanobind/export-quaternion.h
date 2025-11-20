// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <sstream>
#include "../data/Quaternion.h"

namespace nb = nanobind;

namespace spatula { namespace data {

Quaternion make_quaternion(const nb::object& obj) {
    if (!nb::hasattr(obj, "__len__")) {
        throw std::runtime_error("Quaternion object requires a 4 length sequence like object.");
    }
    if (nb::len(obj) < 4) {
        throw std::runtime_error("Quaternion object requires a 4 length sequence like object.");
    }
    nb::tuple t = nb::tuple(obj);
    return Quaternion(nb::cast<double>(t[0]), nb::cast<double>(t[1]), nb::cast<double>(t[2]), nb::cast<double>(t[3]));
}

inline void export_quaternion(nb::module_& m)
{
    nb::class_<Quaternion>(m, "Quaternion")
        .def(nb::init<>())
        .def(nb::init<double, double, double, double>())
        // .def(nb::init(&make_quaternion))
        .def_rw("w", &Quaternion::w)
        .def_rw("x", &Quaternion::x)
        .def_rw("y", &Quaternion::y)
        .def_rw("z", &Quaternion::z)
        .def("conjugate", &Quaternion::conjugate)
        .def("to_axis_angle", &Quaternion::to_axis_angle)
        .def("to_axis_angle_3D", &Quaternion::to_axis_angle_3D)
        .def("norm", &Quaternion::norm)
        .def("normalize", &Quaternion::normalize)
        .def("to_rotation_matrix", &Quaternion::to_rotation_matrix)
        .def("__mul__", [](const Quaternion &a, const Quaternion &b) { return a * b; })
        .def("__imul__", [](Quaternion &a, const Quaternion &b) { a *= b; return a; });
}

}} // namespace spatula::data
