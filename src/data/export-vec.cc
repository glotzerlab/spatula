// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

#include "Vec3.h"
namespace py = pybind11;

namespace spatula { namespace data {

void export_Vec3(py::module& m)
{
    py::class_<Vec3>(m, "Vec3")
        .def(py::init<double, double, double>())
        .def(py::init<>())
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
            py::is_operator())
        .def("__repr__",
             [](const Vec3& v) {
                 auto repr = std::ostringstream();
                 repr << "Vec3(" << std::to_string(v.x) << ", " << std::to_string(v.y) << ", "
                      << std::to_string(v.z) << ")";
                 return repr.str();
             })
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * float())
        .def(py::self / float())
        .def(py::self + float())
        .def(py::self - float())
        .def(
            "__isub__",
            [](Vec3 a, const Vec3 b) { a -= b; },
            py::is_operator())
        .def(
            "__idiv__",
            [](Vec3 a, const Vec3 b) { a /= b; },
            py::is_operator())
        .def(py::self *= py::self)
        .def(py::self += py::self)
        .def(py::self == py::self)
        .def(py::self -= float())
        .def(py::self /= float())
        .def(py::self *= float())
        .def(py::self += float())
        .def(py::self == py::self);
}
}} // namespace spatula::data
