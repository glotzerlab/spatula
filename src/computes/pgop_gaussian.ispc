// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

// ISPC kernel for computing PGOP with Gaussian overlap.
// This kernel implements the fast variant that assumes constant sigmas.

// Define INFINITY for ISPC
#define INFINITY 1e30f

// Fast exponential approximation using degree 5 Remez polynomial
// with Cody-Waite range reduction.
static inline uniform double fast_exp_approx(uniform double x)
{
    uniform double ln2 = 0.69314718055994530941723;
    uniform double ln2_recip = 1.44269504088896340;

    // Compute k = round(x / ln(2))
    uniform double k = round(x * ln2_recip);

    // Cody-Waite range reduction
    uniform double r = x - ln2 * k;

    // Degree 5 Remez approximation using Estrin's method
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

    // Reconstruction: 2^k * p
    // Use pointer casting for type punning
    uniform int64 ki = ((uniform int64)(k + 1023.0)) << 52;
    uniform double scale_factor = *((uniform double *)&ki);

    return p * scale_factor;
}

// Compute the minimum squared distance from symmetrized_position to all positions.
// This is vectorized across the inner loop over positions.
static inline uniform float compute_min_dist_sq(
    uniform const float positions[],
    uniform int32 num_positions,
    uniform float sym_x,
    uniform float sym_y,
    uniform float sym_z)
{
    // Track the minimum squared distance across all lanes
    varying float min_dist_sq = INFINITY;

    // Iterate over positions, vectorized by ISPC
    foreach (m = 0 ... num_positions)
    {
        varying float px = positions[3 * m + 0];
        varying float py = positions[3 * m + 1];
        varying float pz = positions[3 * m + 2];

        varying float dx = px - sym_x;
        varying float dy = py - sym_y;
        varying float dz = pz - sym_z;

        varying float dist_sq = dx * dx + dy * dy + dz * dz;
        min_dist_sq = min(min_dist_sq, dist_sq);
    }

    // Reduce across all lanes to get the global minimum
    return reduce_min(min_dist_sq);
}

// Main kernel: compute PGOP Gaussian fast
// positions: array of Vec3 as flat floats (x0, y0, z0, x1, y1, z1, ...)
// R_ij: array of rotation matrices (9 floats per matrix, row-major)
// num_positions: number of points in positions array
// num_matrices: number of rotation matrices (R_ij.size() / 9)
// sigma: the constant sigma value for all points
export uniform float compute_pgop_gaussian_fast_ispc(
    uniform const float positions[],
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

        // Loop over positions to symmetrize
        for (uniform int32 j = 0; j < num_positions; ++j) {
            // Load the current position
            uniform float px = positions[3 * j + 0];
            uniform float py = positions[3 * j + 1];
            uniform float pz = positions[3 * j + 2];

            // Apply the rotation matrix
            uniform float sym_x = R0 * px + R1 * py + R2 * pz;
            uniform float sym_y = R3 * px + R4 * py + R5 * pz;
            uniform float sym_z = R6 * px + R7 * py + R8 * pz;

            // Find minimum distance from symmetrized position to all positions
            uniform float min_dist_sq = compute_min_dist_sq(
                positions, num_positions, sym_x, sym_y, sym_z);

            // Accumulate the overlap contribution
            overlap += fast_exp_approx(-(uniform double)min_dist_sq * denom);
        }
    }

    // Normalize the result
    uniform float normalization = (uniform float)(num_positions * num_matrices);
    return (uniform float)(overlap / (uniform double)normalization);
}
