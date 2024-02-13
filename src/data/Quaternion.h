#pragma once

#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

#include "Vec3.h"

namespace py = pybind11;

namespace pgop { namespace data {
/**
 * @brief Class provides helper methods for dealing with rotation quaternions.
 *
 * PGOP uses quaternions primarily as an interface to Python to describe points in SO(3).
 * Internally, we use a 3-vector to store the current rotation in optimization, and use rotation
 * matrices to actually perform the rotations. See Util.h for more information on this.
 *
 * We also expose this class to Python to allow for spot-testing of behavior.
 *
 * The unit quaternion for the code's purposes is (1, 0, 0, 0).
 */
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

    /// Return the conjugate of the quaternion (w, -x, -y, -z)
    Quaternion conjugate() const;
    /// Return the norm of the quaterion
    double norm() const;
    /// Normalize the quaternion to a unit quaternion
    void normalize();
    /// Get the recipical of the quaterion q * (-1 / ||q||^2)
    Quaternion recipical() const;
    /// Convert quaternion to a 3x3 rotation matrix
    std::vector<double> to_rotation_matrix() const;
    /// Convert quaternion to its axis angle representation
    std::pair<Vec3, double> to_axis_angle() const;
    /**
     * @brief Convert quaternion to the 3 vector representation
     *
     * The representation adds the angular information into the axis-angle representation by setting
     * the norm of the vector to be the angle.
     */
    Vec3 to_axis_angle_3D() const;

    protected:
    double scale_factor() const;
};

Quaternion operator*(const Quaternion& a, const Quaternion& b);
Quaternion& operator*=(Quaternion& a, const Quaternion& b);

void export_quaternion(py::module& m);
}} // namespace pgop::data
