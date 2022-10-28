#include <cmath>

#include "Quaternion.h"

namespace pgop { namespace data {
Quaternion::Quaternion() : w(1.0), x(0.0), y(0.0), z(0.0) { }

Quaternion::Quaternion(double w_, double x_, double y_, double z_) : w(w_), x(x_), y(y_), z(z_) { }

Quaternion::Quaternion(Vec3 axis, double angle)
{
    axis.normalize();
    const double half_angle = 0.5 * angle;
    w = std::cos(half_angle);
    const double sin_half_angle = std::sin(half_angle);
    x = sin_half_angle * axis.x;
    y = sin_half_angle * axis.y;
    z = sin_half_angle * axis.z;
}

Quaternion Quaternion::conjugate() const
{
    return Quaternion(w, -x, -y, -z);
}

std::vector<double> Quaternion::to_rotation_matrix() const
{
    // Necessary if not unit quaternion. Otherwise it is just 2 / 1 = 2.
    const double s = 2 / (w * w + x * x + y * y + z * z);
    const double xs {x * s}, ys {y * s}, zs {z * s};
    const double wx {w * xs}, wy {w * ys}, wz {w * zs}, xx {x * xs}, xy {x * ys}, xz {x * zs},
        yy {y * ys}, yz {y * zs}, zz {z * zs};
    return std::vector<double> {1 - yy - zz,
                                xy - wz,
                                xz + wy,
                                xy + wz,
                                1 - xx - zz,
                                yz - wx,
                                xz - wy,
                                yz + wx,
                                1 - xx - yy};
}

std::pair<Vec3, double> Quaternion::to_axis_angle() const
{
    const double half_angle = std::acos(w);
    const double sin_qw = half_angle != 0 ? 1 / std::sin(half_angle) : 0;
    return std::make_pair<Vec3, double>({x * sin_qw, y * sin_qw, z * sin_qw}, 2 * half_angle);
}

Vec3 quat_to_vec3(const Quaternion& q)
{
    return Vec3(q.x, q.y, q.z);
}

Quaternion quat_from_hypersphere(double phi, double theta, double psi)
{
    const double sin_phi = std::sin(phi);
    const double sin_psi = std::sin(psi);
    return Quaternion(std::cos(phi),
                      sin_phi * std::cos(psi),
                      sin_phi * sin_psi * std::cos(theta),
                      sin_phi * sin_psi * std::sin(theta));
}

Quaternion quat_from_vec(const Vec3& v)
{
    return Quaternion(0, v.x, v.y, v.z);
}
}} // namespace pgop::data
