#include <cmath>
#include <sstream>
#include <string>

#include <pybind11/operators.h>
#include <pybind11/stl.h>

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

Quaternion::Quaternion(Vec3 v) : Quaternion(v, v.norm()) { }

Quaternion::Quaternion(const py::object& obj)
{
    if (!py::hasattr(obj, "__len__")) {
        throw std::runtime_error("Quaternion object requires a 4 length sequence like object.");
    }
    if (py::len(obj) < 4) {
        throw std::runtime_error("Quaternion object requires a 4 length sequence like object.");
    }
    py::tuple t = py::tuple(obj);
    w = t[0].cast<double>();
    x = t[1].cast<double>();
    y = t[2].cast<double>();
    z = t[3].cast<double>();
}

Quaternion Quaternion::conjugate() const
{
    return Quaternion(w, -x, -y, -z);
}

double Quaternion::norm() const
{
    return std::sqrt(w * w + x * x + y * y + z * z);
}

void Quaternion::normalize()
{
    const double n = norm();
    if (n == 0) {
        return;
    }
    const double inv_norm = 1 / n;
    w *= inv_norm;
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;
}

Quaternion Quaternion::recipical() const
{
    const auto norm_ = norm();
    const auto neg_inv_norm_sq = -1 / (norm_ * norm_);
    return Quaternion(w * neg_inv_norm_sq,
                      x * neg_inv_norm_sq,
                      y * neg_inv_norm_sq,
                      z * neg_inv_norm_sq);
}

std::vector<double> Quaternion::to_rotation_matrix() const
{
    // Necessary if not unit quaternion. Otherwise it is just 2 / 1 = 2.
    const double denominator = w * w + x * x + y * y + z * z;
    if (denominator == 0) {
        return std::vector<double> {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    }
    const double s = 2 / denominator;
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

Vec3 Quaternion::to_axis_angle_3D() const
{
    const auto axis_angle = to_axis_angle();
    return axis_angle.first * axis_angle.second;
}

Quaternion operator*(const Quaternion& a, const Quaternion& b)
{
    return Quaternion(a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
                      a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                      a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                      a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w);
}

Quaternion& operator*=(Quaternion& a, const Quaternion& b)
{
    a = a * b;
    return a;
}

void export_quaternion(py::module& m)
{
    py::class_<Quaternion>(m, "Quaternion")
        .def(py::init<const py::object&>())
        .def_readwrite("w", &Quaternion::w)
        .def_readwrite("x", &Quaternion::x)
        .def_readwrite("y", &Quaternion::y)
        .def_readwrite("z", &Quaternion::z)
        .def("__repr__",
             [](const Quaternion& q) {
                 auto repr = std::ostringstream();
                 repr << "Quaternion(" << std::to_string(q.w) << ", " << std::to_string(q.x) << ", "
                      << std::to_string(q.y) << ", " << std::to_string(q.z) << ")";
                 return repr.str();
             })
        .def("recipical", &Quaternion::recipical)
        .def("conjugate", &Quaternion::conjugate)
        .def("to_axis_angle", &Quaternion::to_axis_angle)
        .def("to_axis_angle_3D", &Quaternion::to_axis_angle_3D)
        .def("norm", &Quaternion::norm)
        .def("normalize", &Quaternion::normalize)
        .def("to_rotation_matrix", &Quaternion::to_rotation_matrix)
        .def(py::self * py::self)
        .def(py::self *= py::self);
}
}} // namespace pgop::data
