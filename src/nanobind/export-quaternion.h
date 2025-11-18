// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <nanobind/nanobind.h>

namespace spatula { namespace data {
void export_quaternion(nanobind::module_& m);
}} // namespace spatula::data
