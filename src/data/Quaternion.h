#pragma once

#include <utility>
#include <vector>

#include "Vec3.h"

namespace pgop { namespace data {
struct Quaternion {
    double w;
    double x;
    double y;
    double z;

    Quaternion();
    Quaternion(double w_, double x_, double y_, double z_);
    Quaternion(Vec3 axis, double angle);

    Quaternion conjugate() const;
    void normalize();
    friend Quaternion quat_from_hypersphere(double phi, double theta, double psi);
    friend Quaternion quat_from_vec(const Vec3& v);
    std::vector<double> to_rotation_matrix() const;
    std::pair<Vec3, double> to_axis_angle() const;

    protected:
    double scale_factor() const;
};

Quaternion operator*(const Quaternion& a, const Quaternion& b);

Quaternion quat_from_hypersphere(double phi, double theta, double psi);
Quaternion quat_from_vec(const Vec3& v);
Vec3 quat_to_vec3(const Quaternion& q);
}} // namespace pgop::data
