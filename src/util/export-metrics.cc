#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/complex.h>

#include "Metrics.h"

namespace nb = nanobind;

void export_Metrics(nb::module_ &m) {
    m.def("covariance", &spatula::util::covariance,
          "Compute the Pearson correlation between two spherical harmonic expansions.");
}
