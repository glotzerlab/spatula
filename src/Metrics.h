#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::array_t<double> covariance(py::array_t<std::complex<double>> qlms,
                               py::array_t<std::complex<double>> sym_qlms);

void export_metrics(py::module& m);
