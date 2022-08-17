#pragma once
#include <complex>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "BondOrder.h"

namespace py = pybind11;

class QlmEval {
    public:
    QlmEval(unsigned int m,
            const py::array_t<double> quad_theta,
            const py::array_t<double> quad_phi,
            const py::array_t<double> weights,
            const py::array_t<std::complex<double>> ylms);

    template<typename distribution>
    py::array_t<std::complex<double>> eval(std::shared_ptr<BondOrder<distribution>> bod);

    private:
    std::vector<double> m_sin_theta;
    std::vector<double> m_cos_theta;
    std::vector<double> m_phi;
    std::vector<std::complex<double>> m_weighted_ylms;
};

void export_qlm_eval(py::module& m);
