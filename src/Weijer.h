#pragma once

#include <complex>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::array_t<std::complex<double>> symmetrize_qlms(py::array_t<std::complex<double>>,
                                                  py::array_t<std::complex<double>>,
                                                  unsigned int max_l);

void export_weijer(py::module& m);
