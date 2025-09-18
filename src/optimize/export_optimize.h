// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace spatula { namespace optimize {
void export_optimize(py::module& m);
}} // namespace spatula::optimize
