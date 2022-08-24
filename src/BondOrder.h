#pragma once

#include <memory>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Null class to show the necessary interface for a distribution.
class SphereDistribution {
    public:
    SphereDistribution();

    double operator()(double theta) const;
};

class UniformDistribution {
    public:
    using param_type = double;

    UniformDistribution(double max_theta);

    double operator()(double theta) const;

    private:
    double m_max_theta;
    double m_prefactor;
};

class FisherDistribution {
    public:
    using param_type = double;

    FisherDistribution(double kappa);

    double operator()(double theta) const;

    private:
    double m_kappa;
    double m_prefactor;
};

template<typename distribution_type> class BondOrder {
    public:
    // Assumes points are on the unit sphere (i.e. normalized)
    BondOrder(distribution_type dist, const py::array_t<double> positions);

    // Assumes points are on the unit sphere (i.e. normalized)
    BondOrder(distribution_type dist, const std::vector<double>& positions);

    double single_call(const double* point) const;

    // Assumes points are on the unit sphere
    py::array_t<double> py_call(const py::array_t<double> points) const;

    std::vector<double> operator()(const std::vector<double>& points) const;

    private:
    distribution_type m_dist;
    std::vector<double> m_positions;
    double m_normalization;
};

template<typename distribution_type> void export_bond_order_class(py::module& m, std::string name);

void export_bond_order(py::module& m);
