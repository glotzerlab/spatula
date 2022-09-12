#include <cmath>

#include "Quaternion.h"

Quaternion::Quaternion(double w_, double x_, double y_, double z_) : w(w_), x(x_), y(y_), z(z_) { }

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

Vec3 quat_to_vec3(const Quaternion& q)
{
    return Vec3(q.x, q.y, q.z);
}

Quaternion quat_from_hypersphere(double phi, double theta, double psi)
{
    const double sin_phi = std::sin(phi);
    const double cos_psi = std::sin(psi);
    return Quaternion(-std::cos(phi),
                      -sin_phi * std::sin(psi),
                      sin_phi * cos_psi * std::sin(theta),
                      -sin_phi * cos_psi * std::cos(theta));
}

Quaternion quat_from_vec(const Vec3& v)
{
    return Quaternion(0, v.x, v.y, v.z);
}
