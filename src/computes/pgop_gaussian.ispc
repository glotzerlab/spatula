// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

// ISPC kernel for computing PGOP with Gaussian overlap.
// This kernel implements the fast variant that assumes constant sigmas.

// Define INFINITY for ISPC
#define INFINITY 1e30f

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

// Positions are passed in Structure-of-Arrays (SoA) format for efficient vector loads.
// pos_x, pos_y, pos_z: separate arrays of position components
// R_ij: array of rotation matrices (9 floats per matrix, row-major)
// num_positions: number of points in positions arrays
// num_matrices: number of rotation matrices (R_ij.size() / 9)
// sigma: the constant sigma value for all points
export uniform float compute_pgop_gaussian_fast_ispc(
    uniform const float pos_x[],
    uniform const float pos_y[],
    uniform const float pos_z[],
    uniform const float R_ij[],
    uniform int32 num_positions,
    uniform int32 num_matrices,
    uniform float sigma)
{
    // Precompute denominator for the exponential
    uniform double denom = 1.0 / (8.0 * (uniform double)sigma * (uniform double)sigma);
    uniform double overlap = 0.0;

    // Loop over each rotation matrix
    for (uniform int32 i = 0; i < num_matrices; ++i) {
        // Load the rotation matrix elements into uniform variables
        uniform float R0 = R_ij[i * 9 + 0];
        uniform float R1 = R_ij[i * 9 + 1];
        uniform float R2 = R_ij[i * 9 + 2];
        uniform float R3 = R_ij[i * 9 + 3];
        uniform float R4 = R_ij[i * 9 + 4];
        uniform float R5 = R_ij[i * 9 + 5];
        uniform float R6 = R_ij[i * 9 + 6];
        uniform float R7 = R_ij[i * 9 + 7];
        uniform float R8 = R_ij[i * 9 + 8];

        // Vectorize over positions to symmetrize - process multiple in parallel
        foreach (j = 0 ... num_positions) {
            // Load the current position from SoA arrays (varying per lane)
            varying float px = pos_x[j];
            varying float py = pos_y[j];
            varying float pz = pos_z[j];

            // Apply the rotation matrix (each lane computes its symmetrized position)
            varying float sym_x = R0 * px + R1 * py + R2 * pz;
            varying float sym_y = R3 * px + R4 * py + R5 * pz;
            varying float sym_z = R6 * px + R7 * py + R8 * pz;

            // Find minimum distance from symmetrized position to all positions
            varying float min_dist_sq = INFINITY;

            for (uniform int32 m = 0; m < num_positions; ++m) {
                uniform float p_x = pos_x[m];
                uniform float p_y = pos_y[m];
                uniform float p_z = pos_z[m];

                varying float dx = p_x - sym_x;
                varying float dy = p_y - sym_y;
                varying float dz = p_z - sym_z;

                varying float dist_sq = dx * dx + dy * dy + dz * dz;
                min_dist_sq = min(min_dist_sq, dist_sq);
            }

            // Accumulate the overlap contribution from all lanes using reduce_add
            varying double exp_val = fast_exp_approx(-min_dist_sq * denom);
            overlap += reduce_add(exp_val);
        }
    }

    // Normalize the result
    uniform float normalization = (uniform float)(num_positions * num_matrices);
    return (uniform float)(overlap / (uniform double)normalization);
}
