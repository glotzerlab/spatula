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
    Mesh(const double* points, size_t n_points) : Optimizer(), m_points()
    {
        m_points.reserve(n_points);
        for (size_t i = 0; i < n_points; ++i)
        {
            const data::Quaternion q(points[i * 4 + 0],
                                     points[i * 4 + 1],
                                     points[i * 4 + 2],
                                     points[i * 4 + 3]);
            m_points.push_back(q.to_axis_angle_3D());
        }
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
