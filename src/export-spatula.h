// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#ifndef SPATULA_EXPORT_SPATULA_H
#define SPATULA_EXPORT_SPATULA_H

#include "pybind11/pybind11.h"

namespace spatula {

void export_BOOSOP(pybind11::module& m);
void export_PGOP(pybind11::module& m);

} // namespace spatula

#endif // SPATULA_EXPORT_SPATULA_H
