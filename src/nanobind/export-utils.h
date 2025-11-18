// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace spatula { namespace util {
void export_util(nb::module_& m);
}} // namespace spatula::util
