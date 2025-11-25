// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/vector.h>

#include "../util/Metrics.h"
#include "export-boosop.h"
#include "export-optimize.h"
#include "export-pgop.h"
#include "export-threads.h"

namespace nb = nanobind;

NB_MODULE(_spatula_nb, m)
{
    m.doc() = "nanobind module for spatula metrics"; // module docstring

    m.def("covariance",
          &spatula::util::covariance,
          "Compute the Pearson correlation between two spherical harmonic expansions.");

    spatula::export_BOOSOP(m);
    spatula::export_spatula(m);
    spatula::util::export_threads(m);
    spatula::optimize::export_optimize(m);
}
