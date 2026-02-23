// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

// Common math functions for ISPC kernels.

// Fast exponential approximation using degree 5 Remez polynomial
// with Cody-Waite range reduction.
// Works with both uniform and varying double inputs.
static inline double fast_exp_approx(double x)
{
    uniform double ln2 = 0.69314718055994530941723;
    uniform double ln2_recip = 1.44269504088896340;

    // Compute k = round(x / ln(2))
    double k = round(x * ln2_recip);

    // Cody-Waite range reduction
    double r = x - ln2 * k;

    // Degree 5 Remez approximation using Estrin's method
    uniform double g = 1.0000000716546679769;
    uniform double f = 0.99999969199097560324;
    uniform double d_const = 0.4999889485139416001;
    uniform double c = 0.16667574730852952047;
    uniform double b = 4.191538198120380032e-2;
    uniform double a = 8.2976549459683138915e-3;

    double r_sq = r * r;
    double p0 = f * r + g;
    double p1 = c * r + d_const;
    double p2 = a * r + b;
    double p1_2 = p2 * r_sq + p1;
    double p = p1_2 * r_sq + p0;

    // Reconstruction: 2^k * p using ldexp
    return ldexp(p, (int32)k);
}

// Uniform approximate exp for scalar inputs
static inline uniform double fast_exp_approx_uniform(uniform double x)
{
    uniform double ln2 = 0.69314718055994530941723;
    uniform double ln2_recip = 1.44269504088896340;

    uniform double k = round(x * ln2_recip);
    uniform double r = x - ln2 * k;

    uniform double g = 1.0000000716546679769;
    uniform double f = 0.99999969199097560324;
    uniform double d_const = 0.4999889485139416001;
    uniform double c = 0.16667574730852952047;
    uniform double b = 4.191538198120380032e-2;
    uniform double a = 8.2976549459683138915e-3;

    uniform double r_sq = r * r;
    uniform double p0 = f * r + g;
    uniform double p1 = c * r + d_const;
    uniform double p2 = a * r + b;
    uniform double p1_2 = p2 * r_sq + p1;
    uniform double p = p1_2 * r_sq + p0;

    return ldexp(p, (uniform int32)k);
}

// Fast hyperbolic sine approximation
// sinh(x) = (e^x - e^-x) / 2
// Works with both uniform and varying double inputs.
static inline double fast_sinh(double x)
{
    double exp_x = fast_exp_approx(x);
    double exp_neg_x = fast_exp_approx(-x);
    return (exp_x - exp_neg_x) * 0.5;
}

// Uniform version for scalar inputs
static inline uniform double fast_sinh_uniform(uniform double x)
{
    uniform double exp_x = fast_exp_approx_uniform(x);
    uniform double exp_neg_x = fast_exp_approx_uniform(-x);
    return (exp_x - exp_neg_x) * 0.5;
}

// Compute the Bhattacharyya coefficient for two Gaussian distributions.
// position: first position vector
// symmetrized_position: second position vector (symmetrized by rotation)
// sigma: standard deviation for first distribution
// sigma_symmetrized: standard deviation for second distribution
static inline double compute_Bhattacharyya_coefficient_gaussian(
    double pos_x, double pos_y, double pos_z,
    double sym_x, double sym_y, double sym_z,
    double sigma, double sigma_symmetrized)
{
    // Compute distance between the two vectors
    double dx = sym_x - pos_x;
    double dy = sym_y - pos_y;
    double dz = sym_z - pos_z;
    double r_sq = dx * dx + dy * dy + dz * dz;

    double sigma_sq = sigma * sigma;
    double sigma_sym_sq = sigma_symmetrized * sigma_symmetrized;
    double sigmas_squared_summed = sigma_sq + sigma_sym_sq;

    double lead_term = (2.0 * sigma * sigma_symmetrized) / sigmas_squared_summed;
    return lead_term * sqrt(lead_term) * fast_exp_approx(-r_sq / (4.0 * sigmas_squared_summed));
}

// Compute the Bhattacharyya coefficient for two Fisher (von Mises-Fisher) distributions.
// Assumes positions are normalized (unit vectors).
// position: first position vector (unit vector)
// symmetrized_position: second position vector (unit vector, symmetrized by rotation)
// kappa: concentration parameter for first distribution
// kappa_symmetrized: concentration parameter for second distribution
static inline double compute_Bhattacharyya_coefficient_fisher_normalized(
    double pos_x, double pos_y, double pos_z,
    double sym_x, double sym_y, double sym_z,
    double kappa, double kappa_symmetrized)
{
    double k1_sq = kappa * kappa;
    double k2_sq = kappa_symmetrized * kappa_symmetrized;
    double k1k2 = kappa * kappa_symmetrized;

    // Dot product of position and symmetrized_position
    double proj = pos_x * sym_x + pos_y * sym_y + pos_z * sym_z;

    double inner_sqrt = k1_sq + k2_sq + 2.0 * k1k2 * proj;
    double sqrt_inner = sqrt(inner_sqrt);

    double sinh_kappa = fast_sinh(kappa);
    double sinh_kappa_sym = fast_sinh(kappa_symmetrized);

    return 2.0 * sqrt(k1k2 / (sinh_kappa * sinh_kappa_sym))
           * fast_sinh(sqrt_inner / 2.0) / sqrt_inner;
}
