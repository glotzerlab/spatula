#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>

#include "Mesh.h"

namespace pgop { namespace optimize {
Mesh::Mesh(const std::vector<data::Quaternion>& points) : Optimizer(), m_points()
{
    m_points.reserve(m_points.size());
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

void export_mesh(nb::module_& m)
{
    nb::class_<Mesh, Optimizer>(m, "Mesh").def(
        nb::init<const std::vector<data::Quaternion>&>());
}
}} // end namespace pgop::optimize
