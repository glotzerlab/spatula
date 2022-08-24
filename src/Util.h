#pragma once

#include <vector>

struct Vec3 {
    double x;
    double y;
    double z;

    Vec3 operator+(Vec3& b);
    Vec3 operator*(Vec3& b);
    Vec3 operator-(Vec3& b);
    Vec3 operator/(Vec3& b);
};

double central_angle(double ref_theta, double ref_phi, double theta, double phi);

double fast_central_angle(double sin_ref_theta,
                          double cos_ref_theta,
                          double ref_phi,
                          double sin_theta,
                          double cos_theta,
                          double phi);

double fast_angle_eucledian(const double* ref_x, const double* x);

void project_to_sphere(const double* x, double* theta, double* phi);

void single_rotate(const double* x, double* x_prime, const std::vector<double>& R);

void rotate_euler(std::vector<double>& rotated_points,
                  const double* x,
                  double alpha,
                  double beta,
                  double gamma);

void rotate_euler(std::vector<double>& rotated_points,
                  const double* x,
                  const std::vector<double>& rotation);

std::vector<double> compute_rotation_matrix(double alpha, double beta, double gamma);

std::vector<double> compute_rotation_matrix(const std::vector<double>& rotation);
