// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>

#include "Vec3.h"

namespace nb = nanobind;

namespace spatula { namespace data {

void export_Vec3(nb::module_& m)
{
    nb::class_<Vec3>(m, "Vec3")
        .def(nb::init<double, double, double>())
        .def(nb::init<>())
        .def("norm", &Vec3::norm)
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z)
        .def("normalize", &Vec3::normalize)
        .def("dot", &Vec3::dot)
        .def("cross", &Vec3::cross)
        .def(
            "__getitem__",
            [](const Vec3& v, size_t i) { return v[i]; },
            nb::is_operator())
        .def("__repr__",
             [](const Vec3& v) {
                 auto repr = std::ostringstream();
                 repr << "Vec3(" << std::to_string(v.x) << ", " << std::to_string(v.y) << ", "
                      << std::to_string(v.z) << ")";
                 return repr.str();
             })
        .def(nb::self * nb::self)
        .def(nb::self / nb::self)
        .def(nb::self + nb::self)
        .def(nb::self - nb::self)
        .def(nb::self * float())
        .def(nb::self / float())
        .def(nb::self + float())
        .def(nb::self - float())
        .def(
            "__isub__",
            [](Vec3 a, const Vec3 b) { a -= b; },
            nb::is_operator())
        .def(
            "__idiv__",
            [](Vec3 a, const Vec3 b) { a /= b; },
            nb::is_operator())
        .def(nb::self *= nb::self)
        .def(nb::self += nb::self)
        .def(nb::self == nb::self)
        .def(nb::self -= float())
        .def(nb::self /= float())
        .def(nb::self *= float())
        .def(nb::self += float())
        .def(nb::self == nb::self);
}
}} // namespace spatula::data
