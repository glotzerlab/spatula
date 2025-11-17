// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>

#include "Mesh.h"

namespace spatula { namespace optimize {
Mesh::Mesh(const std::vector<data::Quaternion>& points) : Optimizer(), m_points()
{
    m_points.reserve(points.size());
    std::transform(points.cbegin(), points.cend(), std::back_inserter(m_points), [](const auto& q) {
        return q.to_axis_angle_3D();
    });
}

void Mesh::internal_next_point()
{
    m_point = m_points[std::min(m_points.size(), static_cast<size_t>(m_count))];
}

bool Mesh::terminate() const
{
    return m_count >= m_points.size();
}

std::unique_ptr<Optimizer> Mesh::clone() const
{
    return std::make_unique<Mesh>(*this);
}
}} // end namespace spatula::optimize
