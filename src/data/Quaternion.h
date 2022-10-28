#pragma once

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

    Quaternion conjugate() const;
    friend Quaternion quat_from_hypersphere(double phi, double theta, double psi);
    friend Quaternion quat_from_vec(const Vec3& v);
    std::vector<double> to_rotation_matrix() const;

    protected:
    double scale_factor() const;
};

Quaternion quat_from_hypersphere(double phi, double theta, double psi);
Quaternion quat_from_vec(const Vec3& v);
Vec3 quat_to_vec3(const Quaternion& q);
}} // namespace pgop::data
