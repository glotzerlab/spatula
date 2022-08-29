#pragma once
#include <complex>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "BondOrder.h"
#include "Util.h"

namespace py = pybind11;

class QlmEval {
    public:
    QlmEval(unsigned int m,
            const py::array_t<double> positions,
            const py::array_t<double> weights,
            const py::array_t<std::complex<double>> ylms);

    template<typename distribution_type>
    std::vector<std::complex<double>> eval(const BondOrder<distribution_type>& bod) const;

    unsigned int getNlm() const;

    private:
    unsigned int m_n_lms;
    unsigned int m_n_points;
    std::vector<Vec3> m_positions;
    std::vector<std::vector<std::complex<double>>> m_weighted_ylms;
};
