#pragma once

#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Null class to show the necessary interface for a distribution.
class SphereDistribution {
    public:
    SphereDistribution();

    double operator()(double theta);
};

class UniformDistribution {
    public:
    UniformDistribution(double max_theta);

    double operator()(double theta);

    private:
    double m_max_theta;
    double m_prefactor;
};

class FisherDistribution {
    public:
    FisherDistribution(double kappa);

    double operator()(double theta);

    private:
    double m_kappa;
    double m_prefactor;
};

template<typename distribution_type> class BondOrder {
    public:
    BondOrder(distribution_type dist, py::array_t<double> theta, py::array_t<double> phi);

    py::array_t<double> operator()(py::array_t<double> theta, py::array_t<double> phi);

    py::array_t<double> fast_call(py::array_t<double> sin_theta,
                                  py::array_t<double> cos_theta,
                                  py::array_t<double> phi);
    double single_call(double sin_theta, double cos_theta, double phi);

    private:
    distribution_type m_dist;
    std::vector<double> m_phi;
    std::vector<double> m_sin_theta;
    std::vector<double> m_cos_theta;
};

template<typename distribution_type> void export_bond_order_class(py::module& m, std::string name);

void export_bond_order(py::module& m);
