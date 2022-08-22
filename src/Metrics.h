#pragma once

#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class WeightedPNormBase {
    public:
    WeightedPNormBase(std::vector<double>& weights);

    virtual double operator()(py::array_t<double> vector) = 0;

    protected:
    std::vector<double> m_weights;
    double m_normalization;
};

template<unsigned int p> class WeightedPNorm : public WeightedPNormBase {
    public:
    WeightedPNorm(std::vector<double>& weights);

    double operator()(py::array_t<double> vector);
};

py::array_t<double> covariance(py::array_t<std::complex<double>> qlms,
                               py::array_t<std::complex<double>> sym_qlms);

void export_pnorm(py::module& m, unsigned int p);

void export_metrics(py::module& m);
