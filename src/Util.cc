#include <cmath>

#include <pybind11/pybind11.h>

#include "Util.h"

double central_angle(double ref_theta, double ref_phi, double theta, double phi)
{
    return fast_central_angle(std::sin(ref_theta),
                              std::cos(ref_theta),
                              ref_phi,
                              std::sin(theta),
                              std::cos(theta),
                              phi);
}

double fast_central_angle(double sin_ref_theta,
                          double cos_ref_theta,
                          double ref_phi,
                          double sin_theta,
                          double cos_theta,
                          double phi)
{
    return std::acos(sin_ref_theta * sin_theta
                     + cos_ref_theta * cos_theta * std::cos(std::abs(ref_phi - phi)));
}

// Assumes points are on the unit sphere
double fast_angle_eucledian(const double* ref_x, const double* x)
{
    const auto dot = ref_x[0] * x[0] + ref_x[1] * x[1] + ref_x[2] * x[2];
    return std::acos(dot);
}

// Assumes points are on the unit sphere
void project_to_sphere(const double* x, double* theta, double* phi)
{
    *theta = std::acos(x[2]);
    // atan2 takes account of what quaterant (x, y) is in necessary for this
    // projection.
    *phi = std::atan2(x[1], x[0]);
}

std::vector<double> compute_rotation_matrix(double alpha, double beta, double gamma)
{
    auto s1 {std::sin(alpha)}, s2 {std::sin(beta)}, s3 {std::sin(gamma)};
    auto c1 {std::cos(alpha)}, c2 {std::cos(beta)}, c3 {std::cos(gamma)};
    auto R = std::vector<double> {
        c2 * c3,
        -c2 * s3,
        s2,
        s1 * s2 * c3 + s3 * c1,
        c1 * c3 - s1 * s2 * s3,
        -s1 * c2,
        s1 * s3 - s2 * c1 * c3,
        s2 * s3 * c1 + c3 * s1,
        c1 * c2,
    };
    return R;
}

std::vector<double> compute_rotation_matrix(const std::vector<double> rotation)
{
    return compute_rotation_matrix(rotation[0], rotation[1], rotation[2]);
}

void single_rotate(const double* x, double* x_prime, const std::vector<double>& R)
{
    x_prime[0] = R[0] * x[0] + R[1] * x[1] + R[2] * x[2];
    x_prime[1] = R[3] * x[0] + R[4] * x[1] + R[5] * x[2];
    x_prime[2] = R[6] * x[0] + R[7] * x[1] + R[8] * x[2];
};

void rotate_euler(std::vector<double>& rotated_points,
                  const double* x,
                  double alpha,
                  double beta,
                  double gamma)
{
    const auto R = compute_rotation_matrix(alpha, beta, gamma);
    const size_t N_points = rotated_points.size() / 3;
    for (size_t i {0}; i < N_points; ++i) {
        single_rotate(&x[i * 3], &rotated_points[i * 3], R);
    }
}

void rotate_euler(std::vector<double>& rotated_points,
                  const double* x,
                  const std::vector<double>& rotation)
{
    return rotate_euler(rotated_points, x, rotation[0], rotation[1], rotation[2]);
}
