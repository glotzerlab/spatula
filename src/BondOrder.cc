#include <cmath>
#include <math.h>

#include "BondOrder.h"
#include "Util.h"

FisherDistribution::FisherDistribution(double kappa)
    : m_kappa(kappa), m_prefactor(kappa / (2 * M_PI * (std::exp(kappa) - std::exp(-kappa))))
{
}

double FisherDistribution::operator()(double theta)
{
    return m_prefactor * std::exp(m_kappa * std::cos(theta));
}

UniformDistribution::UniformDistribution(double max_theta)
    : m_max_theta(max_theta), m_prefactor(1 / (2 * M_PI * (1 - std::cos(max_theta))))
{
}

double UniformDistribution::operator()(double theta)
{
    return theta <= m_max_theta ? m_prefactor : 0;
}

template<typename distribution_type>
BondOrder<distribution_type>::BondOrder(distribution_type dist,
                                        py::array_t<double> theta,
                                        py::array_t<double> phi)
    : m_dist(dist), m_phi(), m_sin_theta(), m_cos_theta()
{
    const auto u_theta = theta.unchecked<1>();
    m_sin_theta.reserve(u_theta.size());
    m_cos_theta.reserve(u_theta.size());
    m_phi.reserve(u_theta.size());
    const auto u_phi = phi.unchecked<1>();
    for (size_t i {0}; i < u_theta.size(); ++i) {
        auto th = u_theta[i] - M_PI_2;
        m_sin_theta.emplace_back(std::sin(th));
        m_cos_theta.emplace_back(std::cos(th));
        m_phi.emplace_back(u_phi[i]);
    }
}

template<typename distribution_type>
py::array_t<double> BondOrder<distribution_type>::operator()(py::array_t<double> theta,
                                                             py::array_t<double> phi)
{
    const auto* u_theta = static_cast<const double*>(theta.data());
    const auto* u_phi = static_cast<const double*>(phi.data());

    auto bo = py::array_t<double>(theta.size());
    auto* u_bo = static_cast<double*>(bo.mutable_data());
    for (size_t i {0}; i < theta.size(); ++i) {
        const double shifted_theta = u_theta[i] - M_PI_2;
        const double sin_theta = std::sin(shifted_theta);
        const double cos_theta = std::cos(shifted_theta);
        u_bo[i] = single_call(sin_theta, cos_theta, u_phi[i]);
    }
    return bo;
}

template<typename distribution_type>
double BondOrder<distribution_type>::single_call(double sin_theta, double cos_theta, double phi)
{
    double value {0};
    for (size_t i {0}; i < m_sin_theta.size(); ++i) {
        const auto angle = fast_central_angle(m_sin_theta[i],
                                              m_cos_theta[i],
                                              m_phi[i],
                                              sin_theta,
                                              cos_theta,
                                              phi);
        value += m_dist(angle);
    }
    return value / static_cast<double>(m_sin_theta.size());
}

template<typename distribution_type>
py::array_t<double> BondOrder<distribution_type>::fast_call(py::array_t<double> sin_theta,
                                                            py::array_t<double> cos_theta,
                                                            py::array_t<double> phi)
{
    const auto* u_sin_theta = static_cast<const double*>(sin_theta.data());
    const auto* u_cos_theta = static_cast<const double*>(cos_theta.data());
    const auto* u_phi = static_cast<const double*>(phi.data());

    auto bo = py::array_t<double>(sin_theta.size());
    auto* u_bo = static_cast<double*>(bo.mutable_data());
    for (size_t i {0}; i < sin_theta.size(); ++i) {
        u_bo[i] = this->single_call(u_sin_theta[i], u_cos_theta[i], u_phi[i]);
    }
    return bo;
}

template<typename distribution_type> void export_bond_order_class(py::module& m, std::string name)
{
    py::class_<BondOrder<distribution_type>>(m, name.c_str())
        .def(py::init<distribution_type, py::array_t<double>, py::array_t<double>>())
        .def("__call__", (&BondOrder<distribution_type>::operator()))
        .def("fast_call", &BondOrder<distribution_type>::fast_call);
}

void export_bond_order(py::module& m)
{
    py::class_<FisherDistribution>(m, "FisherDistribution")
        .def(py::init<double>())
        .def("__call__", &FisherDistribution::operator());

    py::class_<UniformDistribution>(m, "UniformDistribution")
        .def(py::init<double>())
        .def("__call__", &UniformDistribution::operator());

    export_bond_order_class<FisherDistribution>(m, "FisherBondOrder");
    export_bond_order_class<UniformDistribution>(m, "UniformBondOrder");
}
