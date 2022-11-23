#include <cmath>
#include <limits>

#include <pybind11/stl.h>

#include "Mesh.h"

namespace pgop { namespace optimize {
Mesh::Mesh(const std::vector<data::Quaternion>& points) : Optimizer(), m_points(points) { }

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

void export_mesh(py::module& m)
{
    py::class_<Mesh, Optimizer, std::shared_ptr<Mesh>>(m, "Mesh").def(
        py::init<const std::vector<data::Quaternion>&>());
}
}} // end namespace pgop::optimize
