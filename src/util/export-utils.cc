// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Util.h"

namespace py = pybind11;

namespace spatula { namespace util {
void export_util(py::module& m)
{
    m.def("to_rotation_matrix", &to_rotation_matrix);
    m.def("single_rotate", &single_rotate);
}
}} // namespace spatula::util
