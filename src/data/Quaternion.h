#pragma once

#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

#include "Vec3.h"

namespace py = pybind11;

namespace pgop { namespace data {
struct Quaternion {
    double w;
    double x;
    double y;
    double z;

    Quaternion();
    Quaternion(const py::object& obj);
    Quaternion(double w_, double x_, double y_, double z_);
    Quaternion(Vec3 axis, double angle);
    Quaternion(Vec3 axis);

    Quaternion conjugate() const;
    double norm() const;
    void normalize();
    Quaternion recipical() const;
    std::vector<double> to_rotation_matrix() const;
    std::pair<Vec3, double> to_axis_angle() const;
    Vec3 to_axis_angle_3D() const;

    protected:
    double scale_factor() const;
};

Quaternion operator*(const Quaternion& a, const Quaternion& b);
Quaternion& operator*=(Quaternion& a, const Quaternion& b);

void export_quaternion(py::module& m);
}} // namespace pgop::data
