// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#ifndef UTIL_EXPORT_UTILS_H
#define UTIL_EXPORT_UTILS_H

#include <pybind11/pybind11.h>

namespace spatula { namespace util {
void export_util(pybind11::module& m);
}} // namespace spatula::util

#endif // UTIL_EXPORT_UTILS_H
