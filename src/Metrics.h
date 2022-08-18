#pragma once

#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template<unsigned int p> class WeightedPNorm {
    public:
    WeightedPNorm(std::vector<double> weights);

    double operator()(py::array_t<double> vector);

    private:
    std::vector<double> m_weights;
    double m_normalization;
};

py::array_t<double> covariance(py::array_t<std::complex<double>> qlms,
                               py::array_t<std::complex<double>> sym_qlms);

void export_pnorm(py::module& m, unsigned int p);

void export_metrics(py::module& m);
