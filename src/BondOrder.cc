#include <algorithm>
#include <cmath>
#include <math.h>
#include <numeric>

#include "BondOrder.h"
#include "util/Util.h"

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

ApproxLinearDistribution::ApproxLinearDistribution(double alpha)
    : m_prefactor {2.0 / std::sin(alpha)}, m_shift {std::cos(alpha)}
{
}

double ApproxLinearDistribution::operator()(double x) const
{
    return x < m_shift ? 0 : m_prefactor * (x - m_shift);
}

// Todo if no neighbors exist this will lead to nans.
template<typename distribution_type>
BondOrder<distribution_type>::BondOrder(distribution_type dist,
                                        const std::vector<data::Vec3>& positions,
                                        const std::vector<double>& weights)
    : m_dist(dist), m_positions(positions), m_weights(weights),
      m_normalization(1 / std::reduce(m_weights.begin(), m_weights.end()))
{
}

template<typename distribution_type>
double BondOrder<distribution_type>::single_call(const data::Vec3& point) const
{
    double sum_correction = 0;
    // Get the unweighted contribution from each distribution lazily.
    auto single_contributions = std::vector<double>();
    single_contributions.resize(m_positions.size());
    std::transform(m_positions.cbegin(),
                   m_positions.cend(),
                   single_contributions.begin(),
                   [this, &point](const auto& p) -> double {
                       if constexpr (distribution_type::use_theta) {
                           return this->m_dist(util::fast_angle_eucledian(p, point));
                       } else {
                           return this->m_dist(p.dot(point));
                       }
                   });
    // Normalize the value and weight the contributions.
    return m_normalization
           * std::transform_reduce(
               single_contributions.begin(),
               single_contributions.end(),
               m_weights.begin(),
               0.0,
               // Use Kahan summation to improve accuracy of the summation of small
               // numbers.,
               [&sum_correction](const auto& sum, const auto& y) -> double {
                   auto addition = y - sum_correction;
                   const auto new_sum = sum + addition;
                   sum_correction = new_sum - sum - addition;
                   return new_sum;
               },
               std::multiplies<>());
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
template class BondOrder<ApproxLinearDistribution>;
} // End namespace pgop
