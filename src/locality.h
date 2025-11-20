// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <cmath>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "data/Quaternion.h"
#include "data/Vec3.h"
#include "util/Util.h" // For normalize_distances and to_rotation_matrix, rotate_matrix

namespace spatula {

// From BOOSOP.cc
class LocalNeighborhoodBOOBOO
{
  public:
    LocalNeighborhoodBOOBOO(std::vector<data::Vec3>&& positions_, std::vector<double>&& weights_);

    void rotate(const data::Vec3& v);

    std::vector<data::Vec3> positions;
    std::vector<double> weights;
    std::vector<data::Vec3> rotated_positions;
};

class NeighborhoodBOOs
{
  public:
    NeighborhoodBOOs(size_t N,
                     const int* neighbor_counts,
                     const double* weights,
                     const double* distance);

    LocalNeighborhoodBOOBOO getNeighborhoodBOO(size_t i) const;
    std::vector<data::Vec3> getNormalizedDistances(size_t i) const;
    std::vector<double> getWeights(size_t i) const;
    int getNeighborCount(size_t i) const;

  private:
    const size_t m_N;
    const int* m_neighbor_counts;
    const double* m_distances;
    const double* m_weights;
    std::vector<size_t> m_neighbor_offsets;
};

// From PGOP.cc
class LocalNeighborhood
{
  public:
    LocalNeighborhood(std::vector<data::Vec3>&& positions_,
                      std::vector<double>&& weights_,
                      std::vector<double>&& sigmas_);

    void rotate(const data::Vec3& v);

    std::vector<data::Vec3> positions;
    std::vector<double> weights;
    std::vector<double> sigmas;
    std::vector<data::Vec3> rotated_positions;
};

class Neighborhoods
{
  public:
    Neighborhoods(size_t N,
                  const int* neighbor_counts,
                  const double* weights,
                  const double* distance,
                  const double* sigmas);

    LocalNeighborhood getNeighborhood(size_t i) const;
    std::vector<double> getWeights(size_t i) const;
    std::vector<double> getSigmas(size_t i) const;
    int getNeighborCount(size_t i) const;

  private:
    const size_t m_N;
    const int* m_neighbor_counts;
    const double* m_distances;
    const double* m_weights;
    const double* m_sigmas;
    std::vector<size_t> m_neighbor_offsets;
};

// Implementations for LocalNeighborhoodBOOBOO
inline LocalNeighborhoodBOOBOO::LocalNeighborhoodBOOBOO(std::vector<data::Vec3>&& positions_,
                                                        std::vector<double>&& weights_)
    : positions(positions_), weights(weights_), rotated_positions(positions)
{
}

inline void LocalNeighborhoodBOOBOO::rotate(const data::Vec3& v)
{
    const auto R = util::to_rotation_matrix(v);
    util::rotate_matrix(positions.cbegin(), positions.cend(), rotated_positions.begin(), R);
}

// Implementations for NeighborhoodBOOs
inline NeighborhoodBOOs::NeighborhoodBOOs(size_t N,
                                          const int* neighbor_counts,
                                          const double* weights,
                                          const double* distance)
    : m_N {N}, m_neighbor_counts {neighbor_counts}, m_distances {distance}, m_weights {weights},
      m_neighbor_offsets()
{
    m_neighbor_offsets.reserve(m_N + 1);
    m_neighbor_offsets.emplace_back(0);
    std::partial_sum(m_neighbor_counts,
                     m_neighbor_counts + m_N,
                     std::back_inserter(m_neighbor_offsets));
}

inline LocalNeighborhoodBOOBOO NeighborhoodBOOs::getNeighborhoodBOO(size_t i) const
{
    const size_t start {m_neighbor_offsets[i]}, end {m_neighbor_offsets[i + 1]};
    return LocalNeighborhoodBOOBOO(
        util::normalize_distances(m_distances, std::make_pair(3 * start, 3 * end)),
        std::vector(m_weights + start, m_weights + end));
}

inline std::vector<data::Vec3> NeighborhoodBOOs::getNormalizedDistances(size_t i) const
{
    const size_t start {3 * m_neighbor_offsets[i]}, end {3 * m_neighbor_offsets[i + 1]};
    return util::normalize_distances(m_distances, std::make_pair(start, end));
}

inline std::vector<double> NeighborhoodBOOs::getWeights(size_t i) const
{
    const size_t start {m_neighbor_offsets[i]}, end {m_neighbor_offsets[i + 1]};
    return std::vector(m_weights + start, m_weights + end);
}

inline int NeighborhoodBOOs::getNeighborCount(size_t i) const
{
    return m_neighbor_counts[i];
}

// Implementations for LocalNeighborhood
inline LocalNeighborhood::LocalNeighborhood(std::vector<data::Vec3>&& positions_,
                                            std::vector<double>&& weights_,
                                            std::vector<double>&& sigmas_)
    : positions(positions_), weights(weights_), sigmas(sigmas_), rotated_positions(positions)
{
}

inline void LocalNeighborhood::rotate(const data::Vec3& v)
{
    const auto R = util::to_rotation_matrix(v);
    util::rotate_matrix(positions.cbegin(), positions.cend(), rotated_positions.begin(), R);
}

// Implementations for Neighborhoods
inline Neighborhoods::Neighborhoods(size_t N,
                                    const int* neighbor_counts,
                                    const double* weights,
                                    const double* distance,
                                    const double* sigmas)
    : m_N {N}, m_neighbor_counts {neighbor_counts}, m_distances {distance}, m_weights {weights},
      m_sigmas {sigmas}, m_neighbor_offsets()
{
    m_neighbor_offsets.reserve(m_N + 1);
    m_neighbor_offsets.emplace_back(0);
    std::partial_sum(m_neighbor_counts,
                     m_neighbor_counts + m_N,
                     std::back_inserter(m_neighbor_offsets));
}

inline LocalNeighborhood Neighborhoods::getNeighborhood(size_t i) const
{
    const size_t start {m_neighbor_offsets[i]}, end {m_neighbor_offsets[i + 1]};

    // Create a vector of Vec3 to store the positions (3 coordinates for each Vec3)
    std::vector<data::Vec3> neighborhood_positions;
    neighborhood_positions.reserve(end - start);

    for (size_t j = start; j < end; ++j) {
        // Each Vec3 contains 3 consecutive elements from m_distances
        neighborhood_positions.emplace_back(
            data::Vec3 {m_distances[3 * j], m_distances[3 * j + 1], m_distances[3 * j + 2]});
    }

    return LocalNeighborhood(std::move(neighborhood_positions),
                             std::vector(m_weights + start, m_weights + end),
                             std::vector(m_sigmas + start, m_sigmas + end));
}

inline std::vector<double> Neighborhoods::getWeights(size_t i) const
{
    const size_t start {m_neighbor_offsets[i]}, end {m_neighbor_offsets[i + 1]};
    return std::vector(m_weights + start, m_weights + end);
}

inline std::vector<double> Neighborhoods::getSigmas(size_t i) const
{
    const size_t start {m_neighbor_offsets[i]}, end {m_neighbor_offsets[i + 1]};
    return std::vector(m_sigmas + start, m_sigmas + end);
}

inline int Neighborhoods::getNeighborCount(size_t i) const
{
    return m_neighbor_counts[i];
}

// Global functions from PGOP.cc
inline double compute_Bhattacharyya_coefficient_gaussian(const data::Vec3& position,
                                                         const data::Vec3& symmetrized_position,
                                                         double sigma,
                                                         double sigma_symmetrized)
{
    // 1. compute the distance between the two vectors (symmetrized_position
    //    and positions[m])
    auto r_pos = symmetrized_position - position;
    auto sigmas_squared_summed = sigma * sigma + sigma_symmetrized * sigma_symmetrized;
    // 2. compute the gaussian overlap between the two points. Bhattacharyya coefficient
    //    is used.
    return std::pow((2 * sigma * sigma_symmetrized / sigmas_squared_summed), 3 / 2)
           * std::exp(-r_pos.dot(r_pos) / (4 * sigmas_squared_summed));
}

inline double compute_Bhattacharyya_coefficient_fisher(const data::Vec3& position,
                                                       const data::Vec3& symmetrized_position,
                                                       double kappa,
                                                       double kappa_symmetrized)
{
    auto position_norm = std::sqrt(position.dot(position));
    auto symmetrized_position_norm = std::sqrt(symmetrized_position.dot(symmetrized_position));
    // If position norm is zero vector means this point is at origin and contributes 1
    // to the overlap, check that with a small epsilon.
    if ((position_norm < 1e-10) && (symmetrized_position_norm < 1e-10)) {
        return 1;
    } else if ((position_norm < 1e-10) || (symmetrized_position_norm < 1e-10)) {
        return 0;
    }
    auto k1_sq = kappa * kappa;
    auto k2_sq = kappa_symmetrized * kappa_symmetrized;
    auto k1k2 = kappa * kappa_symmetrized;
    auto proj = position.dot(symmetrized_position) / (position_norm * symmetrized_position_norm);
    return 2 * std::sqrt(k1k2 / (std::sinh(kappa) * std::sinh(kappa_symmetrized)))
           * std::sinh((std::sqrt(k1_sq + k2_sq + 2 * k1k2 * proj)) / 2)
           / std::sqrt(k1_sq + k2_sq + 2 * k1k2 * proj);
}

} // namespace spatula