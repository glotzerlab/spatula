// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include "export-utils.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>

#include "../data/Vec3.h"
#include "../util/Util.h"

namespace nb = nanobind;

namespace spatula { namespace util {
void export_util(nb::module_& m)
{
    m.def("to_rotation_matrix", &to_rotation_matrix,
          nb::arg("v"),
          "Convert a Vec3 representing an axis-angle rotation to a rotation "
          "matrix.");

    m.def("single_rotate",
          [](const Vec3& x, const std::vector<double>& R) {
              Vec3 x_prime;
              single_rotate(x, x_prime, R);
              return x_prime;
          },
          nb::arg("x"), nb::arg("R"), "Rotate a point by a rotation matrix.");
}
}} // namespace spatula::util
