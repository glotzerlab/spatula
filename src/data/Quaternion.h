// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <cmath>
#include <utility>

#include "RotationMatrix.h"
#include "Vec3.h"

namespace spatula { namespace data {
/**
 * @brief Class provides helper methods for dealing with rotation quaternions.
 *
 * spatula uses quaternions primarily as an interface to Python to describe points in SO(3).
 * Internally, we use a 3-vector to store the current rotation in optimization, and use rotation
 * matrices to actually perform the rotations. See Util.h for more information on this.
 *
 * We also expose this class to Python to allow for spot-testing of behavior.
 *
 * The unit quaternion for the code's purposes is (1, 0, 0, 0).
 */
struct Quaternion {
    float w;
    float x;
    float y;
    float z;

    Quaternion() : w(1.0f), x(0.0f), y(0.0f), z(0.0f) { }
    Quaternion(float w_, float x_, float y_, float z_) : w(w_), x(x_), y(y_), z(z_) { }
    Quaternion(Vec3 axis, float angle)
    {
        axis.normalize();
        const float half_angle = 0.5f * angle;
        w = std::cos(half_angle);
        const float sin_half_angle = std::sin(half_angle);
        x = sin_half_angle * axis.x;
        y = sin_half_angle * axis.y;
        z = sin_half_angle * axis.z;
    }
    Quaternion(Vec3 v) : Quaternion(v, static_cast<float>(v.norm())) { }

    /// Return the conjugate of the quaternion (w, -x, -y, -z)
    Quaternion conjugate() const
    {
        return Quaternion(w, -x, -y, -z);
    }
    /// Return the norm of the quaterion
    float norm() const
    {
        return std::sqrt(w * w + x * x + y * y + z * z);
    }
    /// Normalize the quaternion to a unit quaternion
    void normalize()
    {
        const float n = norm();
        if (n == 0) {
            return;
        }
        const float inv_norm = 1.0f / n;
        w *= inv_norm;
        x *= inv_norm;
        y *= inv_norm;
        z *= inv_norm;
    }
    /// Convert quaternion to a 3x3 rotation matrix
    RotationMatrix to_rotation_matrix() const
    {
        // Necessary if not unit quaternion. Otherwise it is just 2 / 1 = 2.
        const float denominator = w * w + x * x + y * y + z * z;
        if (denominator == 0) {
            return RotationMatrix {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        }
        const float s = 2.0f / denominator;
        const float xs {x * s}, ys {y * s}, zs {z * s};
        const float wx {w * xs}, wy {w * ys}, wz {w * zs}, xx {x * xs}, xy {x * ys}, xz {x * zs},
            yy {y * ys}, yz {y * zs}, zz {z * zs};
        return RotationMatrix {1.0f - yy - zz,
                               xy - wz,
                               xz + wy,
                               xy + wz,
                               1.0f - xx - zz,
                               yz - wx,
                               xz - wy,
                               yz + wx,
                               1.0f - xx - yy};
    }
    /// Convert quaternion to its axis angle representation
    std::pair<Vec3, float> to_axis_angle() const
    {
        const float half_angle = std::acos(w);
        const float sin_qw = half_angle != 0 ? 1.0f / std::sin(half_angle) : 0;
        return std::make_pair<Vec3, float>({x * sin_qw, y * sin_qw, z * sin_qw}, 2.0f * half_angle);
    }
    /**
     * @brief Convert quaternion to the 3 vector representation
     *
     * The representation adds the angular information into the axis-angle representation by setting
     * the norm of the vector to be the angle.
     */
    Vec3 to_axis_angle_3D() const
    {
        const auto axis_angle = to_axis_angle();
        return axis_angle.first * axis_angle.second;
    }
};

inline Quaternion operator*(const Quaternion& a, const Quaternion& b)
{
    return Quaternion(a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
                      a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                      a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                      a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w);
}

inline Quaternion& operator*=(Quaternion& a, const Quaternion& b)
{
    a = a * b;
    return a;
}

}} // namespace spatula::data
