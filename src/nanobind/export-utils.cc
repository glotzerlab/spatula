// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include "export-utils.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/vector.h>

#include "../data/Vec3.h"
#include "../util/Metrics.h"
#include "../util/QlmEval.h"
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

    m.def("covariance",
          &spatula::util::covariance,
          "Compute the Pearson correlation between two spherical harmonic expansions.");

    nb::class_<QlmBuf>(m, "QlmBuf")
        .def(nb::init<size_t>())
        .def_readwrite("qlms", &QlmBuf::qlms)
        .def_readwrite("sym_qlms", &QlmBuf::sym_qlms);

    nb::class_<QlmEval>(m, "QlmEval")
        .def(nb::init<unsigned int,
                      const nb::ndarray<double, nb::shape<nb::any, 3>, nb::c_contig>,
                      const nb::ndarray<double, nb::shape<nb::any>, nb::c_contig>,
                      const nb::ndarray<std::complex<double>, nb::shape<nb::any, nb::any>, nb::c_contig>>(),
             nb::arg("m"), nb::arg("positions"), nb::arg("weights"),
             nb::arg("ylms"))
        .def("eval",
             static_cast<std::vector<std::complex<double>> (QlmEval::*)(
                 const BondOrder<UniformDistribution>&) const>(&QlmEval::eval<UniformDistribution>),
             nb::arg("bod"),
             "Evaluate the Qlms for a given bond order diagram.")
        .def("eval",
             static_cast<std::vector<std::complex<double>> (QlmEval::*)(
                 const BondOrder<FisherDistribution>&) const>(&QlmEval::eval<FisherDistribution>),
             nb::arg("bod"),
             "Evaluate the Qlms for a given bond order diagram.")
        .def("getNlm", &QlmEval::getNlm, "Get the number of lm values.")
        .def("getMaxL", &QlmEval::getMaxL, "Get the maximum l value.");
}
}} // namespace spatula::util
