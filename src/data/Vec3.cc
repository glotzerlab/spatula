#include "Vec3.h"

namespace pgop { namespace data {
Vec3::Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) { }

Vec3::Vec3(const double* point) : x(point[0]), y(point[1]), z(point[2]) { }

Vec3::Vec3() : x(0.0), y(0.0), z(0.0) { }

double Vec3::dot(const Vec3& b) const
{
    return x * b.x + y * b.y + z * b.z;
}

template<typename number_type> Vec3 operator+(const Vec3& a, const number_type& b)
{
    return Vec3(a.x + b, a.y + b, a.z + b);
}

template<typename number_type> Vec3 operator-(const Vec3& a, const number_type& b)
{
    return Vec3(a.x - b, a.y - b, a.z - b);
}

template<typename number_type> Vec3 operator*(const Vec3& a, const number_type& b)
{
    return Vec3(a.x * b, a.y * b, a.z * b);
}

template<typename number_type> Vec3 operator/(const Vec3& a, const number_type& b)
{
    return Vec3(a.x / b, a.y / b, a.z / b);
}

template<> Vec3 operator+(const Vec3& a, const Vec3& b)
{
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

template<> Vec3 operator-(const Vec3& a, const Vec3& b)
{
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

template<> Vec3 operator*(const Vec3& a, const Vec3& b)
{
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

template<> Vec3 operator/(const Vec3& a, const Vec3& b)
{
    return Vec3(a.x / b.x, a.y / b.y, a.z / b.z);
}

template<typename number_type> Vec3& operator+=(Vec3& a, const number_type& b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    return a;
}

template<typename number_type> Vec3& operator-=(Vec3& a, const number_type& b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    return a;
}

template<typename number_type> Vec3& operator*=(Vec3& a, const number_type& b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

template<typename number_type> Vec3& operator/=(Vec3& a, const number_type& b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    return a;
}

template<> Vec3& operator+=(Vec3& a, const Vec3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

template<> Vec3& operator-=(Vec3& a, const Vec3& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

template<> Vec3& operator*=(Vec3& a, const Vec3& b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}

template<> Vec3& operator/=(Vec3& a, const Vec3& b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    return a;
}

template Vec3 operator+(const Vec3& a, const double& b);
template Vec3 operator-(const Vec3& a, const double& b);
template Vec3 operator*(const Vec3& a, const double& b);
template Vec3 operator/(const Vec3& a, const double& b);
template Vec3& operator+=(Vec3& a, const double& b);
template Vec3& operator-=(Vec3& a, const double& b);
template Vec3& operator*=(Vec3& a, const double& b);
template Vec3& operator/=(Vec3& a, const double& b);
}} // namespace pgop::data
