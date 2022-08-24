#include <cmath>
#include <math.h>
#include <memory>

#include "BondOrder.h"
#include "Util.h"

FisherDistribution::FisherDistribution(double kappa)
    : m_kappa(kappa), m_prefactor(kappa / (2 * M_PI * (std::exp(kappa) - std::exp(-kappa))))
{
}

double FisherDistribution::operator()(double theta) const
{
    return m_prefactor * std::exp(m_kappa * std::cos(theta));
}

UniformDistribution::UniformDistribution(double max_theta)
    : m_max_theta(max_theta), m_prefactor(1 / (2 * M_PI * (1 - std::cos(max_theta))))
{
}

double UniformDistribution::operator()(double theta) const
{
    return theta <= m_max_theta ? m_prefactor : 0;
}

template<typename distribution_type>
BondOrder<distribution_type>::BondOrder(distribution_type dist, const py::array_t<double> positions)
    : m_dist(dist), m_positions(), m_normalization(1 / static_cast<double>(positions.shape(0)))
{
    const auto u_positions = positions.unchecked<2>();
    m_positions.assign(u_positions.data(0, 0), u_positions.data(0, 0) + u_positions.size());
}

template<typename distribution_type>
BondOrder<distribution_type>::BondOrder(distribution_type dist,
                                        const std::vector<double>& positions)
    : m_dist(dist), m_positions(positions),
      m_normalization(3 / static_cast<double>(positions.size()))
{
}

template<typename distribution_type>
py::array_t<double> BondOrder<distribution_type>::py_call(const py::array_t<double> points) const
{
    auto u_points = points.unchecked<2>();
    auto bo = py::array_t<double>(u_points.shape(0));
    auto u_bo = static_cast<double*>(bo.mutable_data(0));
    for (size_t i {0}; i < u_points.shape(0); ++i) {
        u_bo[i] = this->single_call(points.data(i, 0));
    }
    return bo;
}

template<typename distribution_type>
double BondOrder<distribution_type>::single_call(const double* point) const
{
    double value {0};
    for (size_t i {0}; i < m_positions.size(); i += 3) {
        const auto angle = fast_angle_eucledian(&m_positions[0] + i, point);
        value += m_dist(angle);
    }
    return m_normalization * value;
}

template<typename distribution_type>
std::vector<double>
BondOrder<distribution_type>::operator()(const std::vector<double>& points) const
{
    auto bo = std::vector<double>();
    const size_t n_points = points.size() / 3;
    bo.reserve(n_points);
    for (size_t i {0}; i < points.size(); i += 3) {
        bo.push_back(this->single_call(&points[i]));
    }
    return bo;
}

template<typename distribution_type> void export_bond_order_class(py::module& m, std::string name)
{
    py::class_<BondOrder<distribution_type>, std::shared_ptr<BondOrder<distribution_type>>>(
        m,
        name.c_str())
        .def(py::init<distribution_type, py::array_t<double>>())
        .def("__call__", &BondOrder<distribution_type>::py_call);
}

// explicitly create templates
template class BondOrder<UniformDistribution>;
template class BondOrder<FisherDistribution>;

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
