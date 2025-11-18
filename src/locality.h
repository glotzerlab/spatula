// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <numeric>
#include <vector>

#include "data/Vec3.h"
#include "util/Util.h"

namespace spatula {

struct LocalNeighborhood {
    LocalNeighborhood(std::vector<data::Vec3>&& positions_,
                      std::vector<double>&& weights_,
                      std::vector<double>&& sigmas_ = {})
        : positions(std::move(positions_)), weights(std::move(weights_)),
          sigmas(std::move(sigmas_)), rotated_positions(positions)
    {
    }

    void rotate(const data::Vec3& v)
    {
        const auto R = util::to_rotation_matrix(v);
        util::rotate_matrix(positions.cbegin(), positions.cend(), rotated_positions.begin(), R);
    }

    const std::vector<data::Vec3> positions;
    const std::vector<double> weights;
    const std::vector<double> sigmas;
    std::vector<data::Vec3> rotated_positions;
};

class Neighborhoods {
    public:
    Neighborhoods(size_t N,
                  const int* neighbor_counts,
                  const double* weights,
                  const double* distance,
                  const double* sigmas = nullptr)
        : m_N(N), m_neighbor_counts(neighbor_counts), m_distances(distance), m_weights(weights),
          m_sigmas(sigmas), m_neighbor_offsets()
    {
        m_neighbor_offsets.reserve(m_N + 1);
        m_neighbor_offsets.emplace_back(0);
        std::partial_sum(m_neighbor_counts,
                         m_neighbor_counts + m_N,
                         std::back_inserter(m_neighbor_offsets));
    }

    LocalNeighborhood getNeighborhood(size_t i) const
    {
        const size_t start {m_neighbor_offsets[i]};
        const size_t end {m_neighbor_offsets[i + 1]};
        std::vector<double> sigmas;
        if (m_sigmas) {
            sigmas = std::vector(m_sigmas + start, m_sigmas + end);
        }
        return LocalNeighborhood(
            util::normalize_distances(m_distances, std::make_pair(3 * start, 3 * end)),
            std::vector(m_weights + start, m_weights + end),
            std::move(sigmas));
    }

    std::vector<data::Vec3> getNormalizedDistances(size_t i) const
    {
        const size_t start {3 * m_neighbor_offsets[i]}, end {3 * m_neighbor_offsets[i + 1]};
        return util::normalize_distances(m_distances, std::make_pair(start, end));
    }

    std::vector<double> getWeights(size_t i) const
    {
        const size_t start {m_neighbor_offsets[i]}, end {m_neighbor_offsets[i + 1]};
        return std::vector(m_weights + start, m_weights + end);
    }

    std::vector<double> getSigmas(size_t i) const
    {
        if (!m_sigmas) {
            return {};
        }
        const size_t start {m_neighbor_offsets[i]}, end {m_neighbor_offsets[i + 1]};
        return std::vector(m_sigmas + start, m_sigmas + end);
    }

    int getNeighborCount(size_t i) const
    {
        return m_neighbor_counts[i];
    }

    private:
    const size_t m_N;
    const int* m_neighbor_counts;
    const double* m_distances;
    const double* m_weights;
    const double* m_sigmas;
    std::vector<size_t> m_neighbor_offsets;
};

} // namespace spatula
