// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <sstream>
#include <string>

#include "Quaternion.h"

namespace py = pybind11;

namespace spatula { namespace data {

Quaternion::Quaternion(const py::object& obj)
{
    if (!py::hasattr(obj, "__len__")) {
        throw std::runtime_error("Quaternion object requires a 4 length sequence like object.");
    }
    if (py::len(obj) < 4) {
        throw std::runtime_error("Quaternion object requires a 4 length sequence like object.");
    }
    py::tuple t = py::tuple(obj);
    w = t[0].cast<double>();
    x = t[1].cast<double>();
    y = t[2].cast<double>();
    z = t[3].cast<double>();
}

void export_quaternion(py::module& m)
{
    py::class_<Quaternion>(m, "Quaternion")
        .def(py::init<const py::object&>())
        .def_readwrite("w", &Quaternion::w)
        .def_readwrite("x", &Quaternion::x)
        .def_readwrite("y", &Quaternion::y)
        .def_readwrite("z", &Quaternion::z)
        .def("__repr__",
             [](const Quaternion& q) {
                 auto repr = std::ostringstream();
                 repr << "Quaternion(" << std::to_string(q.w) << ", " << std::to_string(q.x) << ", "
                      << std::to_string(q.y) << ", " << std::to_string(q.z) << ")";
                 return repr.str();
             })
        .def("conjugate", &Quaternion::conjugate)
        .def("to_axis_angle", &Quaternion::to_axis_angle)
        .def("to_axis_angle_3D", &Quaternion::to_axis_angle_3D)
        .def("norm", &Quaternion::norm)
        .def("normalize", &Quaternion::normalize)
        .def("to_rotation_matrix", &Quaternion::to_rotation_matrix)
        .def(py::self * py::self)
        .def(py::self *= py::self);
}
}} // namespace spatula::data
