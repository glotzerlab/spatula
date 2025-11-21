// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <memory>
#include <vector>

#include "../data/Quaternion.h"
#include "../data/Vec3.h"
#include "Optimize.h"

namespace spatula { namespace optimize {

/**
 * @brief An Optimizer that just tests prescribed points.
 *
 * The optimizer picks the best point out of all provided test points.
 */
class Mesh : public Optimizer {
    public:
    /**
     * @brief Create an Mesh.
     *
     * All parameters are expected to have matching dimensions.
     *
     * @param points The points to test. Expected sizes are \f$ (N_{brute}, N_{dim} \f$.
     */
    Mesh(const std::vector<data::Quaternion>& points) : Optimizer(), m_points()
    {
        m_points.reserve(points.size());
        std::transform(points.cbegin(),
                       points.cend(),
                       std::back_inserter(m_points),
                       [](const auto& q) { return q.to_axis_angle_3D(); });
    }

    ~Mesh() override = default;

    void internal_next_point() override
    {
        m_point = m_points[std::min(m_points.size(), static_cast<size_t>(m_count))];
    }

    bool terminate() const override
    {
        return m_count >= m_points.size();
    }

    std::unique_ptr<Optimizer> clone() const override
    {
        return std::make_unique<Mesh>(*this);
    }

    private:
    /// The set of points to evaluate.
    std::vector<data::Vec3> m_points;
};

}} // namespace spatula::optimize
