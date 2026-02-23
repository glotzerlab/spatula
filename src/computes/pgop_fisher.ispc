// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

// ISPC kernel for computing PGOP with Fisher (von Mises-Fisher) overlap.
// This kernel implements the fast variant that assumes constant kappa.

#include "math.ispc"

// Positions are passed in Structure-of-Arrays (SoA) format for efficient vector loads.
// pos_x, pos_y, pos_z: separate arrays of position components (unit vectors on sphere)
// R_ij: array of rotation matrices (9 floats per matrix, row-major)
// num_positions: number of points in positions arrays
// num_matrices: number of rotation matrices (R_ij.size() / 9)
// kappa: the concentration parameter for the von Mises-Fisher distribution
export uniform float compute_pgop_fisher_fast_ispc(
    uniform const float pos_x[],
    uniform const float pos_y[],
    uniform const float pos_z[],
    uniform const float R_ij[],
    uniform int32 num_positions,
    uniform int32 num_matrices,
    uniform float kappa)
{
    // Precompute prefix term (uniform, computed once)
    uniform double prefix_term = 2.0 * (uniform double)kappa / fast_sinh_uniform((uniform double)kappa);
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

            // Find maximum projection from symmetrized position to all positions
            // Clamp lower bound to -1.0 in case of numerical underflow
            varying float max_proj = -1.0f;

            for (uniform int32 m = 0; m < num_positions; ++m) {
                uniform float p_x = pos_x[m];
                uniform float p_y = pos_y[m];
                uniform float p_z = pos_z[m];

                // Dot product: position dot symmetrized_position
                varying float proj = p_x * sym_x + p_y * sym_y + p_z * sym_z;
                max_proj = max(max_proj, proj);
            }

            // Compute inner term: kappa * sqrt(2 * (1 + max_proj))
            varying double inner_term = (uniform double)kappa * sqrt(2.0 * (1.0 + (varying double)max_proj));

            // Compute contribution with singularity handling
            varying double contribution;
            if (inner_term > 1e-6) {
                // Normal case: prefix_term * sinh(inner_term * 0.5) / inner_term
                contribution = prefix_term * fast_sinh(inner_term * 0.5) / inner_term;
            } else {
                // Handle singularity at inner_term near 0 (when max_proj is near -1.0)
                contribution = prefix_term * 0.5;
            }

            // Accumulate the overlap contribution from all lanes using reduce_add
            overlap += reduce_add(contribution);
        }
    }

    // Normalize the result
    const uniform float normalization = (uniform float)(num_positions * num_matrices);
    return (uniform float)(overlap / (uniform double)normalization);
}

// Non-fast version that handles per-point kappas.
// kappas: array of kappa values, one per position
export uniform float compute_pgop_fisher_ispc(
    uniform const float pos_x[],
    uniform const float pos_y[],
    uniform const float pos_z[],
    uniform const float R_ij[],
    uniform const float kappas[],
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
            varying float kappa_j = kappas[j];

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
                uniform float kappa_m = kappas[m];

                varying double bc = compute_Bhattacharyya_coefficient_fisher_normalized(
                    (varying double)p_x, (varying double)p_y, (varying double)p_z,
                    (varying double)sym_x, (varying double)sym_y, (varying double)sym_z,
                    (varying double)kappa_j, (uniform double)kappa_m);

                max_bc = max(max_bc, bc);
            }

            overlap += reduce_add(max_bc);
        }
    }

    const uniform float normalization = (uniform float)(num_positions * num_matrices);
    return (uniform float)(overlap / (uniform double)normalization);
}
