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

template<typename T> class LocalNeighborhood {
    public:
    LocalNeighborhood(std::vector<data::Vec3<T>>&& positions_,
                      std::span<const double> weights_,
                      std::span<const double> sigmas_);
    LocalNeighborhood(std::vector<data::Vec3<T>>&& positions_, std::span<const double> weights_);

    void rotate(const data::Vec3<T>& v);

    bool constantSigmas() const;

    std::vector<data::Vec3<T>> positions;
    std::span<const double> weights;
    std::span<const double> sigmas;
    std::vector<data::Vec3<T>> rotated_positions;

    private:
    bool m_constant_sigmas = false;
};

template<typename T>
inline LocalNeighborhood<T>::LocalNeighborhood(std::vector<data::Vec3<T>>&& positions_,
                                               std::span<const double> weights_,
                                               std::span<const double> sigmas_)
    : positions(positions_), weights(weights_), sigmas(sigmas_), rotated_positions(positions)
{
    // Verify whether all sigma values are equivalent
    m_constant_sigmas
        = !sigmas.empty()
          && (std::adjacent_find(sigmas.begin(), sigmas.end(), std::not_equal_to<double>())
              == sigmas.end());
}

template<typename T>
inline LocalNeighborhood<T>::LocalNeighborhood(std::vector<data::Vec3<T>>&& positions_,
                                               std::span<const double> weights_)
    : positions(positions_), weights(weights_), rotated_positions(positions)
{
    m_constant_sigmas = false;
}

template<typename T> inline bool LocalNeighborhood<T>::constantSigmas() const
{
    return m_constant_sigmas;
}

template<typename T> inline void LocalNeighborhood<T>::rotate(const data::Vec3<T>& v)
{
    const auto R = data::RotationMatrix<T>::from_vec3(v);
    util::rotate_matrix<T>(positions.cbegin(), positions.cend(), rotated_positions.begin(), R);
}

template<typename D>
class Neighborhoods {
    public:
    Neighborhoods(size_t N,
                  const int* neighbor_counts,
                  const double* weights,
                  const D* distance,
                  bool normalize_distances,
                  const double* sigmas = nullptr);

    // Returns LocalNeighborhood with position type matching distance type
    LocalNeighborhood<D> getNeighborhood(size_t i) const;
    std::span<const double> getWeights(size_t i) const;
    std::span<const double> getSigmas(size_t i) const;
    int getNeighborCount(size_t i) const;

    private:
    const size_t m_N;
    const int* m_neighbor_counts;
    const D* m_distances;
    const double* m_weights;
    const double* m_sigmas;
    std::vector<size_t> m_neighbor_offsets;
    bool m_normalize_distances;
};

template<typename D>
inline Neighborhoods<D>::Neighborhoods(size_t N,
                                    const int* neighbor_counts,
                                    const double* weights,
                                    const D* distance,
                                    bool normalize_distances,
                                    const double* sigmas)
    : m_N {N}, m_neighbor_counts {neighbor_counts}, m_distances {distance},
      m_weights {weights}, m_sigmas {sigmas}, m_neighbor_offsets(), m_normalize_distances {normalize_distances}
{
    m_neighbor_offsets.reserve(m_N + 1);
    m_neighbor_offsets.emplace_back(0);
    std::partial_sum(m_neighbor_counts,
                     m_neighbor_counts + m_N,
                     std::back_inserter(m_neighbor_offsets));
}

template<typename D>
inline LocalNeighborhood<D> Neighborhoods<D>::getNeighborhood(size_t i) const
{
    const size_t start {m_neighbor_offsets[i]}, end {m_neighbor_offsets[i + 1]};
    const size_t num_neighbors = end - start;

    std::vector<data::Vec3<D>> neighborhood_positions;
    if (m_normalize_distances) {
        neighborhood_positions
            = util::normalize_distances<D>(m_distances, std::make_pair(3 * start, 3 * end));
    } else {
        neighborhood_positions.reserve(num_neighbors);
        for (size_t j = start; j < end; ++j) {
            neighborhood_positions.emplace_back(
                data::Vec3<D> {static_cast<D>(m_distances[3 * j]),
                                   static_cast<D>(m_distances[3 * j + 1]),
                                   static_cast<D>(m_distances[3 * j + 2])});
        }
    }

    if (m_sigmas) {
        return LocalNeighborhood<D>(std::move(neighborhood_positions),
                                    std::span(m_weights + start, num_neighbors),
                                    std::span(m_sigmas + start, num_neighbors));
    }
    return LocalNeighborhood<D>(std::move(neighborhood_positions),
                                std::span(m_weights + start, num_neighbors));
}

template<typename D>
inline std::span<const double> Neighborhoods<D>::getWeights(size_t i) const
{
    const size_t start {m_neighbor_offsets[i]}, end {m_neighbor_offsets[i + 1]};
    return std::span(m_weights + start, end - start);
}

template<typename D>
inline std::span<const double> Neighborhoods<D>::getSigmas(size_t i) const
{
    const size_t start {m_neighbor_offsets[i]}, end {m_neighbor_offsets[i + 1]};
    if (m_sigmas) {
        return std::span(m_sigmas + start, end - start);
    }
    return std::span<const double>();
}

template<typename D>
inline int Neighborhoods<D>::getNeighborCount(size_t i) const
{
    return m_neighbor_counts[i];
}

// Typedefs for common precision types
using LocalNeighborhoodd = LocalNeighborhood<double>;
using LocalNeighborhoodf = LocalNeighborhood<float>;

} // namespace spatula
