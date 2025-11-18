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

void export_quaternion(py::module& m)
{
    py::class_<Quaternion>(m, "Quaternion")
        .def(py::init<double, double, double, double>())
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
        .def(py::self *= py::self)
        .def_static(
            "from_object",
            [](const py::object& obj) {
                if (!py::hasattr(obj, "__len__")) {
                    throw std::runtime_error("Quaternion object requires a 4 length sequence like object.");
                }
                if (py::len(obj) < 4) {
                    throw std::runtime_error("Quaternion object requires a 4 length sequence like object.");
                }
                py::tuple t = py::tuple(obj);
                return Quaternion(t[0].cast<double>(), t[1].cast<double>(), t[2].cast<double>(), t[3].cast<double>());
            },
            "Create a Quaternion from a 4-element sequence (w, x, y, z).");
}
}} // namespace spatula::data
