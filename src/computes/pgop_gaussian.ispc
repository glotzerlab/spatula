// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

// ISPC kernel for computing PGOP with Gaussian overlap.
// This kernel implements the fast variant that assumes constant sigmas.

#include "math.ispc"

// Define INFINITY for ISPC
#define INFINITY 1e30f

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
        const uniform float R0 = R_ij[i * 9 + 0];
        const uniform float R1 = R_ij[i * 9 + 1];
        const uniform float R2 = R_ij[i * 9 + 2];
        const uniform float R3 = R_ij[i * 9 + 3];
        const uniform float R4 = R_ij[i * 9 + 4];
        const uniform float R5 = R_ij[i * 9 + 5];
        const uniform float R6 = R_ij[i * 9 + 6];
        const uniform float R7 = R_ij[i * 9 + 7];
        const uniform float R8 = R_ij[i * 9 + 8];

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
    const uniform float normalization = (uniform float)(num_positions * num_matrices);
    return (uniform float)(overlap / (uniform double)normalization);
}

// Non-fast version that handles per-point sigmas.
// sigmas: array of sigma values, one per position
export uniform float compute_pgop_gaussian_ispc(
    uniform const float pos_x[],
    uniform const float pos_y[],
    uniform const float pos_z[],
    uniform const float R_ij[],
    uniform const float sigmas[],
    uniform int32 num_positions,
    uniform int32 num_matrices)
{
    uniform double overlap = 0.0;

    // Loop over each rotation matrix
    for (uniform int32 i = 0; i < num_matrices; ++i) {
        // Load the rotation matrix elements
        const uniform float R0 = R_ij[i * 9 + 0];
        const uniform float R1 = R_ij[i * 9 + 1];
        const uniform float R2 = R_ij[i * 9 + 2];
        const uniform float R3 = R_ij[i * 9 + 3];
        const uniform float R4 = R_ij[i * 9 + 4];
        const uniform float R5 = R_ij[i * 9 + 5];
        const uniform float R6 = R_ij[i * 9 + 6];
        const uniform float R7 = R_ij[i * 9 + 7];
        const uniform float R8 = R_ij[i * 9 + 8];

        // Vectorize over positions to symmetrize
        foreach (j = 0 ... num_positions) {
            varying float px = pos_x[j];
            varying float py = pos_y[j];
            varying float pz = pos_z[j];
            varying float sigma_j = sigmas[j];

            // Apply the rotation matrix
            varying float sym_x = R0 * px + R1 * py + R2 * pz;
            varying float sym_y = R3 * px + R4 * py + R5 * pz;
            varying float sym_z = R6 * px + R7 * py + R8 * pz;

            // Find maximum Bhattacharyya coefficient
            varying double max_bc = 0.0;

            for (uniform int32 m = 0; m < num_positions; ++m) {
                uniform float p_x = pos_x[m];
                uniform float p_y = pos_y[m];
                uniform float p_z = pos_z[m];
                uniform float sigma_m = sigmas[m];

                varying double bc = compute_Bhattacharyya_coefficient_gaussian(
                    (varying double)p_x, (varying double)p_y, (varying double)p_z,
                    (varying double)sym_x, (varying double)sym_y, (varying double)sym_z,
                    (varying double)sigma_j, (uniform double)sigma_m);

                max_bc = max(max_bc, bc);
            }

            overlap += reduce_add(max_bc);
        }
    }

    const uniform float normalization = (uniform float)(num_positions * num_matrices);
    return (uniform float)(overlap / (uniform double)normalization);
}
