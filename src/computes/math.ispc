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

// Uniform version for scalar inputs
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
