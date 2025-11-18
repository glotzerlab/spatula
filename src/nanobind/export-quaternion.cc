// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

#include "Quaternion.h"
#include "Vec3.h" // Needed for Vec3 in Quaternion constructor

namespace nb = nanobind;

namespace spatula { namespace data {

void export_quaternion(nb::module_& m)
{
    nb::class_<Quaternion>(m, "Quaternion")
        .def(nb::init<double, double, double, double>())
        .def(nb::init<Vec3, double>())
        .def(nb::init<Vec3>())
        .def(nb::init<const nb::object&>(),
             nb::arg("obj"),
             "Construct a Quaternion from a 4-element sequence like object.")
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
        .def(nb::self * nb::self)
        .def(nb::self *= nb::self);
}

}} // namespace spatula::data
