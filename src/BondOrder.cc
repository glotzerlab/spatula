#include <algorithm>
#include <cmath>
#include <math.h>
#include <numeric>

#include "BondOrder.h"

namespace pgop {
FisherDistribution::FisherDistribution(double kappa)
    : m_kappa(kappa), m_prefactor(kappa / (2 * M_PI * (std::exp(kappa) - std::exp(-kappa))))
{
}

double FisherDistribution::operator()(double x) const
{
    return m_prefactor * std::exp(m_kappa * x);
}

UniformDistribution::UniformDistribution(double max_theta)
    : m_threshold(std::cos(max_theta)), m_prefactor(1 / (2 * M_PI * (1 - std::cos(max_theta))))
{
}

double UniformDistribution::operator()(double x) const
{
    return x > m_threshold ? m_prefactor : 0;
}

template<typename distribution_type>
BondOrder<distribution_type>::BondOrder(distribution_type dist,
                                        const std::vector<data::Vec3>& positions)
    : m_dist(dist), m_positions(positions),
      m_normalization(1 / static_cast<double>(positions.size()))
{
}

template<typename distribution_type>
double BondOrder<distribution_type>::single_call(const data::Vec3& point) const
{
    double sum_correction = 0;
    // Use Kahan summation to improve accuracy of the summation of small
    // numbers.
    return m_normalization
           * std::accumulate(
               m_positions.cbegin(),
               m_positions.cend(),
               0.0,
               [this, &point, &sum_correction](const auto& sum, const auto& p) -> double {
                   double x;
                   if constexpr (distribution_type::use_theta) {
                       x = util::fast_angle_eucledian(p, point);
                   } else {
                       x = p.dot(point);
                   }
                   const auto addition = this->m_dist(x) - sum_correction;
                   const auto new_sum = sum + addition;
                   sum_correction = new_sum - sum - addition;
                   return new_sum;
               });
}

template<typename distribution_type>
std::vector<double>
BondOrder<distribution_type>::operator()(const std::vector<data::Vec3>& points) const
{
    auto bo = std::vector<double>();
    bo.reserve(points.size());
    std::transform(points.cbegin(),
                   points.cend(),
                   std::back_inserter(bo),
                   [this](const auto& point) { return this->single_call(point); });
    return bo;
}

// explicitly create templates
template class BondOrder<UniformDistribution>;
template class BondOrder<FisherDistribution>;
} // End namespace pgop
