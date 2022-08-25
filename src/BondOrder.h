#pragma once

#include <memory>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Util.h"

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
    BondOrder(distribution_type dist, const std::vector<Vec3>& positions);

    double single_call(const Vec3& point) const;

    // Assumes points are on the unit sphere
    std::vector<double> operator()(const std::vector<Vec3>& points) const;

    private:
    distribution_type m_dist;
    std::vector<Vec3> m_positions;
    double m_normalization;
};
