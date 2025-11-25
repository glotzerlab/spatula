// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace spatula {
void export_BOOSOP(py::module& m);
} // namespace spatula