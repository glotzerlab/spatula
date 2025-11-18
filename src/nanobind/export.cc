// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/vector.h>

#include "export-optimizer.h"
#include "export-quat.h" // Include the new header
#include "export-stores.h"
#include "export-threads.h"
#include "export-utils.h"

namespace nb = nanobind;

NB_MODULE(_spatula_nb, m)
{
    m.doc() = "nanobind module for spatula metrics"; // module docstring

    spatula::util::export_threads(m);
    spatula::data::bind_quaternion(m); // Call the bind_quaternion function
    spatula::optimize::export_optimizers(m);
    spatula::util::export_util(m);
    spatula::export_stores(m);
}
