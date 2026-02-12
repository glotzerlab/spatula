// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace spatula {
void export_BOOSOP(nb::module_& m);
} // namespace spatula
