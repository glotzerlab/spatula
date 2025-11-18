// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <nanobind/nanobind.h>

namespace spatula { namespace optimize {
void export_optimize(nanobind::module_& m);
void export_mesh(nanobind::module_& m);
void export_no_optimization(nanobind::module_& m);
void export_base_optimize(nanobind::module_& m);
void export_random_search(nanobind::module_& m);
void export_step_gradient_descent(nanobind::module_& m);
void export_union(nanobind::module_& m);
}} // namespace spatula::optimize
