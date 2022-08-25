#pragma once

#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct Vec3 {
    double x;
    double y;
    double z;

    Vec3(double x, double y, double z);
    Vec3(const double* point);
    Vec3();

    double dot(const Vec3& b) const;
};

template<typename number_type> Vec3 operator+(const Vec3& a, const number_type& b);

template<typename number_type> Vec3 operator-(const Vec3& a, const number_type& b);

template<typename number_type> Vec3 operator*(const Vec3& a, const number_type& b);

template<typename number_type> Vec3 operator/(const Vec3& a, const number_type& b);

template<typename number_type> Vec3& operator+=(Vec3& a, const number_type& b);

template<typename number_type> Vec3& operator-=(Vec3& a, const number_type& b);

template<typename number_type> Vec3& operator*=(Vec3& a, const number_type& b);

template<typename number_type> Vec3& operator/=(Vec3& a, const number_type& b);

double fast_angle_eucledian(const Vec3& ref_x, const Vec3& x);

void single_rotate(const Vec3& x, Vec3& x_prime, const std::vector<double>& R);

void rotate_euler(const std::vector<Vec3>::const_iterator points_begin,
                  const std::vector<Vec3>::const_iterator points_end,
                  std::vector<Vec3>::iterator rotated_points_it,
                  double alpha,
                  double beta,
                  double gamma);

void rotate_euler(std::vector<Vec3>::const_iterator points_begin,
                  std::vector<Vec3>::const_iterator points_end,
                  std::vector<Vec3>::iterator rotated_points_it,
                  const std::vector<double>& rotation);

std::vector<double> compute_rotation_matrix(double alpha, double beta, double gamma);

std::vector<double> compute_rotation_matrix(const std::vector<double>& rotation);

std::vector<Vec3> normalize_distances(const py::array_t<double> distances);
