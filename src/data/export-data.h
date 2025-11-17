// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#ifndef DATA_EXPORT_DATA_H
#define DATA_EXPORT_DATA_H

#include <pybind11/pybind11.h>

namespace spatula { namespace data {
void export_data(pybind11::module& m);
void export_quaternion(pybind11::module& m);
void export_Vec3(pybind11::module& m);
}} // namespace spatula::data

#endif // DATA_EXPORT_DATA_H
