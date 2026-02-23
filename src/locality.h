// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <span>
#include <vector>

#include "data/RotationMatrix.h"
#include "data/Vec3.h"
#include "util/Util.h"

namespace spatula {

class LocalNeighborhood {
    public:
    LocalNeighborhood(std::vector<data::Vec3>&& positions_,
                      std::span<const float> weights_,
                      std::span<const float> sigmas_);
    LocalNeighborhood(std::vector<data::Vec3>&& positions_, std::span<const float> weights_);

    void rotate(const data::Vec3& v);

    bool constantSigmas() const;

    std::vector<data::Vec3> positions;
    std::span<const float> weights;
    std::span<const float> sigmas;
    std::vector<data::Vec3> rotated_positions;

    private:
    bool m_constant_sigmas = false;
};

inline LocalNeighborhood::LocalNeighborhood(std::vector<data::Vec3>&& positions_,
                                            std::span<const float> weights_,
                                            std::span<const float> sigmas_)
    : positions(std::move(positions_)), weights(weights_), sigmas(sigmas_),
      rotated_positions(positions) // The rotated positions must be a copy
{
    // Verify whether all sigma values are equivalent
    m_constant_sigmas
        = !sigmas.empty()
          && (std::adjacent_find(sigmas.begin(), sigmas.end(), std::not_equal_to<double>())
              == sigmas.end());
}

inline LocalNeighborhood::LocalNeighborhood(std::vector<data::Vec3>&& positions_,
                                            std::span<const float> weights_)
    : positions(std::move(positions_)), weights(weights_), rotated_positions(positions)
{
    m_constant_sigmas = false;
}

inline bool LocalNeighborhood::constantSigmas() const
{
    return m_constant_sigmas;
}

inline void LocalNeighborhood::rotate(const data::Vec3& v)
{
    const auto R = data::RotationMatrix::from_vec3(v);
    util::rotate_matrix(positions.cbegin(), positions.cend(), rotated_positions.begin(), R);
}

class Neighborhoods {
    public:
    Neighborhoods(size_t N,
                  const int* neighbor_counts,
                  const float* weights,
                  const float* distance,
                  bool normalize_distances,
                  const float* sigmas = nullptr);

    // Returns LocalNeighborhood with position type matching distance type
    LocalNeighborhood getNeighborhood(size_t i) const;
    std::span<const float> getWeights(size_t i) const;
    std::span<const float> getSigmas(size_t i) const;
    int getNeighborCount(size_t i) const;

    private:
    const size_t m_N;
    const int* m_neighbor_counts;
    const float* m_distances;
    const float* m_weights;
    const float* m_sigmas;
    std::vector<size_t> m_neighbor_offsets;
    bool m_normalize_distances;
};

inline Neighborhoods::Neighborhoods(size_t N,
                                    const int* neighbor_counts,
                                    const float* weights,
                                    const float* distance,
                                    bool normalize_distances,
                                    const float* sigmas)
    : m_N {N}, m_neighbor_counts {neighbor_counts}, m_distances {distance}, m_weights {weights},
      m_sigmas {sigmas}, m_neighbor_offsets(), m_normalize_distances {normalize_distances}
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
    const size_t num_neighbors = end - start;

    std::vector<data::Vec3> neighborhood_positions;
    if (m_normalize_distances) {
        neighborhood_positions
            = util::normalize_distances(m_distances, std::make_pair(3 * start, 3 * end));
    } else {
        neighborhood_positions.reserve(num_neighbors);
        for (size_t j = start; j < end; ++j) {
            neighborhood_positions.emplace_back(
                data::Vec3 {m_distances[3 * j], m_distances[3 * j + 1], m_distances[3 * j + 2]});
        }
    }

    if (m_sigmas) {
        return LocalNeighborhood(std::move(neighborhood_positions),
                                 std::span(m_weights + start, num_neighbors),
                                 std::span(m_sigmas + start, num_neighbors));
    }
    return LocalNeighborhood(std::move(neighborhood_positions),
                             std::span(m_weights + start, num_neighbors));
}

inline std::span<const float> Neighborhoods::getWeights(size_t i) const
{
    const size_t start {m_neighbor_offsets[i]}, end {m_neighbor_offsets[i + 1]};
    return std::span(m_weights + start, end - start);
}

inline std::span<const float> Neighborhoods::getSigmas(size_t i) const
{
    const size_t start {m_neighbor_offsets[i]}, end {m_neighbor_offsets[i + 1]};
    if (m_sigmas) {
        return std::span(m_sigmas + start, end - start);
    }
    return std::span<const float>();
}

inline int Neighborhoods::getNeighborCount(size_t i) const
{
    return m_neighbor_counts[i];
}

} // namespace spatula
