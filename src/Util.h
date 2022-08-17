#pragma once

#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

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

std::vector<double> compute_rotation_matrix(double alpha, double beta, double gamma);

py::array_t<double>
rotate_euler(const py::array_t<double> x, double alpha, double beta, double gamma);

void export_util(py::module& m);
