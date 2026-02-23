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

// Fast hyperbolic sine approximation
// sinh(x) = (e^x - e^-x) / 2
// Works with both uniform and varying double inputs.
static inline double fast_sinh(double x)
{
    double exp_x = fast_exp_approx(x);
    double exp_neg_x = fast_exp_approx(-x);
    return (exp_x - exp_neg_x) * 0.5;
}

// Compute the Bhattacharyya coefficient for two Gaussian distributions.
static inline double compute_Bhattacharyya_coefficient_gaussian(
    double pos_x, double pos_y, double pos_z,
    double sym_x, double sym_y, double sym_z,
    double sigma, double sigma_symmetrized)
{
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
static inline double compute_Bhattacharyya_coefficient_fisher_normalized(
    double pos_x, double pos_y, double pos_z,
    double sym_x, double sym_y, double sym_z,
    double kappa, double kappa_symmetrized)
{
    double k1_sq = kappa * kappa;
    double k2_sq = kappa_symmetrized * kappa_symmetrized;
    double k1k2 = kappa * kappa_symmetrized;

    double proj = pos_x * sym_x + pos_y * sym_y + pos_z * sym_z;

    double inner_sqrt = k1_sq + k2_sq + 2.0 * k1k2 * proj;
    double sqrt_inner = sqrt(inner_sqrt);

    double sinh_kappa = fast_sinh(kappa);
    double sinh_kappa_sym = fast_sinh(kappa_symmetrized);

    return 2.0 * sqrt(k1k2 / (sinh_kappa * sinh_kappa_sym))
           * fast_sinh(sqrt_inner / 2.0) / sqrt_inner;
}

uniform struct RotationMatrix
{
    uniform float R0, R1, R2, R3, R4, R5, R6, R7, R8;
};

static inline uniform RotationMatrix load_rotation_matrix(uniform const float R_ij[], uniform int32 matrix_idx)
{
    uniform RotationMatrix R;
    uniform int32 base = matrix_idx * 9;
    R.R0 = R_ij[base + 0];
    R.R1 = R_ij[base + 1];
    R.R2 = R_ij[base + 2];
    R.R3 = R_ij[base + 3];
    R.R4 = R_ij[base + 4];
    R.R5 = R_ij[base + 5];
    R.R6 = R_ij[base + 6];
    R.R7 = R_ij[base + 7];
    R.R8 = R_ij[base + 8];
    return R;
}

static inline void apply_rotation(
    varying float px, varying float py, varying float pz,
    uniform const RotationMatrix R,
    varying float& sym_x, varying float& sym_y, varying float& sym_z)
{
    sym_x = R.R0 * px + R.R1 * py + R.R2 * pz;
    sym_y = R.R3 * px + R.R4 * py + R.R5 * pz;
    sym_z = R.R6 * px + R.R7 * py + R.R8 * pz;
}

static inline varying float distance_squared(
    varying float x1, varying float y1, varying float z1,
    varying float x2, varying float y2, varying float z2)
{
    varying float dx = x2 - x1;
    varying float dy = y2 - y1;
    varying float dz = z2 - z1;
    return dx * dx + dy * dy + dz * dz;
}

static inline varying float dot_product(
    varying float x1, varying float y1, varying float z1,
    varying float x2, varying float y2, varying float z2)
{
    return x1 * x2 + y1 * y2 + z1 * z2;
}

// Macro for the common PGOP outer loop structure
// Usage: PGOP_OUTER_LOOP(inner_computation_block)
// The inner block must define `varying double contribution`
// Assumes: R_ij, num_matrices, num_positions, pos_x, pos_y, pos_z, overlap are in scope
#define PGOP_OUTER_LOOP(INNER_BLOCK) \
    for (uniform int32 i = 0; i < num_matrices; ++i) { \
        uniform RotationMatrix R = load_rotation_matrix(R_ij, i); \
        foreach (j = 0 ... num_positions) { \
            varying float px = pos_x[j]; \
            varying float py = pos_y[j]; \
            varying float pz = pos_z[j]; \
            varying float sym_x, sym_y, sym_z; \
            apply_rotation(px, py, pz, R, sym_x, sym_y, sym_z); \
            INNER_BLOCK \
            overlap += reduce_add(contribution); \
        } \
    } \
    return (uniform float)(overlap / (uniform double)(num_positions * num_matrices));
