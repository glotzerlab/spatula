// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>

#include "BOOSOP.h"
#include "PGOP.h"
#include "data/Vec3.h"
#include "util/export-utils.h"

PYBIND11_MODULE(_spatula, m)
{
    spatula::export_spatula(m);
    spatula::export_BOOSOP(m);
    spatula::util::export_util(m);
}
