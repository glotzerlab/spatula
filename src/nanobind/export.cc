// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/vector.h>

#include "../util/Metrics.h" // Relative path to Metrics.h
#include "export-optimizer.h"
#include "export-quat.h" // Include the new header
#include "export-threads.h"
#include "export-vec.h"

namespace nb = nanobind;

NB_MODULE(_spatula_nb, m)
{
    m.doc() = "nanobind module for spatula metrics"; // module docstring

    m.def("covariance",
          &spatula::util::covariance,
          "Compute the Pearson correlation between two spherical harmonic expansions.");

    spatula::util::export_threads(m);
    spatula::data::bind_quaternion(m); // Call the bind_quaternion function
    spatula::data::bind_vec3(m);
    spatula::optimize::export_optimizers(m);
}
