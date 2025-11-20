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
#include "util/Util.h"

namespace spatula {

class LocalNeighborhoodBOOBOO {
    public:
    LocalNeighborhoodBOOBOO(std::vector<data::Vec3>&& positions_, std::vector<double>&& weights_);

    void rotate(const data::Vec3& v);

    std::vector<data::Vec3> positions;
    std::vector<double> weights;
    std::vector<data::Vec3> rotated_positions;
};

class NeighborhoodBOOs {
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

class LocalNeighborhood {
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

class Neighborhoods {
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

} // namespace spatula
