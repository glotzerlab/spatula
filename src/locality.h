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
    LocalNeighborhood(size_t max_neighbors); // Pre-allocate for reuse

    void rotate(const data::Vec3& v);
    void reset();

    bool constantSigmas() const;

    // SoA format
    std::vector<float> pos_x, pos_y, pos_z;
    std::vector<float> rotated_pos_x, rotated_pos_y, rotated_pos_z;

    std::span<const float> weights;
    std::span<const float> sigmas;

    private:
    bool m_constant_sigmas = false;
    friend class Neighborhoods;
};

inline LocalNeighborhood::LocalNeighborhood(std::vector<data::Vec3>&& positions_,
                                            std::span<const float> weights_,
                                            std::span<const float> sigmas_)
    : weights(weights_), sigmas(sigmas_)
{
    // Convert to SoA format
    const size_t n = positions_.size();
    pos_x.resize(n);
    pos_y.resize(n);
    pos_z.resize(n);
    rotated_pos_x.resize(n);
    rotated_pos_y.resize(n);
    rotated_pos_z.resize(n);
    for (size_t i = 0; i < n; ++i) {
        pos_x[i] = positions_[i].x;
        pos_y[i] = positions_[i].y;
        pos_z[i] = positions_[i].z;
        rotated_pos_x[i] = positions_[i].x;
        rotated_pos_y[i] = positions_[i].y;
        rotated_pos_z[i] = positions_[i].z;
    }

    // Verify whether all sigma values are equivalent
    m_constant_sigmas
        = !sigmas.empty()
          && (std::adjacent_find(sigmas.begin(), sigmas.end(), std::not_equal_to<double>())
              == sigmas.end());
}

inline LocalNeighborhood::LocalNeighborhood(std::vector<data::Vec3>&& positions_,
                                            std::span<const float> weights_)
    : weights(weights_)
{
    // Convert to SoA format
    const size_t n = positions_.size();
    pos_x.resize(n);
    pos_y.resize(n);
    pos_z.resize(n);
    rotated_pos_x.resize(n);
    rotated_pos_y.resize(n);
    rotated_pos_z.resize(n);
    for (size_t i = 0; i < n; ++i) {
        pos_x[i] = positions_[i].x;
        pos_y[i] = positions_[i].y;
        pos_z[i] = positions_[i].z;
        rotated_pos_x[i] = positions_[i].x;
        rotated_pos_y[i] = positions_[i].y;
        rotated_pos_z[i] = positions_[i].z;
    }

    m_constant_sigmas = false;
}

inline LocalNeighborhood::LocalNeighborhood(size_t max_neighbors)
    : weights {}, sigmas {}, m_constant_sigmas {false}
{
    pos_x.reserve(max_neighbors);
    pos_y.reserve(max_neighbors);
    pos_z.reserve(max_neighbors);
    rotated_pos_x.reserve(max_neighbors);
    rotated_pos_y.reserve(max_neighbors);
    rotated_pos_z.reserve(max_neighbors);
}

inline bool LocalNeighborhood::constantSigmas() const
{
    return m_constant_sigmas;
}

inline void LocalNeighborhood::reset()
{
    std::copy(pos_x.begin(), pos_x.end(), rotated_pos_x.begin());
    std::copy(pos_y.begin(), pos_y.end(), rotated_pos_y.begin());
    std::copy(pos_z.begin(), pos_z.end(), rotated_pos_z.begin());
}

inline void LocalNeighborhood::rotate(const data::Vec3& v)
{
    const auto R = data::RotationMatrix::from_vec3(v);

    const size_t n = pos_x.size();
    for (size_t i = 0; i < n; ++i) {
        rotated_pos_x[i] = R[0] * pos_x[i] + R[1] * pos_y[i] + R[2] * pos_z[i];
        rotated_pos_y[i] = R[3] * pos_x[i] + R[4] * pos_y[i] + R[5] * pos_z[i];
        rotated_pos_z[i] = R[6] * pos_x[i] + R[7] * pos_y[i] + R[8] * pos_z[i];
    }
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
    void fillNeighborhood(size_t i, LocalNeighborhood& neighborhood) const;
    int getMaxNeighborCount() const;
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

inline void Neighborhoods::fillNeighborhood(size_t i, LocalNeighborhood& neighborhood) const
{
    const size_t start {m_neighbor_offsets[i]}, end {m_neighbor_offsets[i + 1]};
    const size_t num_neighbors = end - start;

    neighborhood.pos_x.resize(num_neighbors);
    neighborhood.pos_y.resize(num_neighbors);
    neighborhood.pos_z.resize(num_neighbors);
    neighborhood.rotated_pos_x.resize(num_neighbors);
    neighborhood.rotated_pos_y.resize(num_neighbors);
    neighborhood.rotated_pos_z.resize(num_neighbors);

    if (m_normalize_distances) {
        auto normalized
            = util::normalize_distances(m_distances, std::make_pair(3 * start, 3 * end));
        for (size_t j = 0; j < num_neighbors; ++j) {
            neighborhood.pos_x[j] = normalized[j].x;
            neighborhood.pos_y[j] = normalized[j].y;
            neighborhood.pos_z[j] = normalized[j].z;
            neighborhood.rotated_pos_x[j] = normalized[j].x;
            neighborhood.rotated_pos_y[j] = normalized[j].y;
            neighborhood.rotated_pos_z[j] = normalized[j].z;
        }
    } else {
        for (size_t j = 0; j < num_neighbors; ++j) {
            const float x = m_distances[3 * (start + j)];
            const float y = m_distances[3 * (start + j) + 1];
            const float z = m_distances[3 * (start + j) + 2];
            neighborhood.pos_x[j] = x;
            neighborhood.pos_y[j] = y;
            neighborhood.pos_z[j] = z;
            neighborhood.rotated_pos_x[j] = x;
            neighborhood.rotated_pos_y[j] = y;
            neighborhood.rotated_pos_z[j] = z;
        }
    }

    neighborhood.weights = std::span(m_weights + start, num_neighbors);
    if (m_sigmas) {
        neighborhood.sigmas = std::span(m_sigmas + start, num_neighbors);
        neighborhood.m_constant_sigmas = std::adjacent_find(neighborhood.sigmas.begin(),
                                                            neighborhood.sigmas.end(),
                                                            std::not_equal_to<float>())
                                         == neighborhood.sigmas.end();
    } else {
        neighborhood.sigmas = std::span<const float>();
        neighborhood.m_constant_sigmas = false;
    }
}

inline int Neighborhoods::getMaxNeighborCount() const
{
    return *std::max_element(m_neighbor_counts, m_neighbor_counts + m_N);
}

} // namespace spatula
