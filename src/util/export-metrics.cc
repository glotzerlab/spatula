// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/vector.h>

#include "Metrics.h"

namespace nb = nanobind;

void export_Metrics(nb::module_& m)
{
    m.def("covariance",
          &spatula::util::covariance,
          "Compute the Pearson correlation between two spherical harmonic expansions.");
}
