// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>

#include "BOOSOP.h"
#include "PGOP.h"
#include "data/Quaternion.h"
// #include "data/export-quaternion.cc"

PYBIND11_MODULE(_spatula, m)
{
    // spatula::data::export_quaternion(m);
    spatula::export_spatula(m);
    spatula::export_BOOSOP(m);
}
