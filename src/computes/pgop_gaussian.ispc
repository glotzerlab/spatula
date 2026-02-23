// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

// ISPC kernel for computing PGOP with Gaussian overlap.

#include "math.ispc"

// Define INFINITY for ISPC
#define INFINITY 1e30f

// Fast version: assumes constant sigmas.
export uniform float compute_pgop_gaussian_fast_ispc(
    uniform const float pos_x[],
    uniform const float pos_y[],
    uniform const float pos_z[],
    uniform const float R_ij[],
    uniform int32 num_positions,
    uniform int32 num_matrices,
    uniform float sigma)
{
    uniform double denom = 1.0 / (8.0 * (uniform double)sigma * (uniform double)sigma);
    uniform double overlap = 0.0;

    PGOP_OUTER_LOOP(
        varying float min_dist_sq = INFINITY;
        for (uniform int32 m = 0; m < num_positions; ++m) {
            varying float dist_sq = distance_squared(
                sym_x, sym_y, sym_z,
                (varying float)pos_x[m], (varying float)pos_y[m], (varying float)pos_z[m]);
            min_dist_sq = min(min_dist_sq, dist_sq);
        }
        varying double contribution = fast_exp_approx(-min_dist_sq * denom);
    )
}

// Non-fast version: handles per-point sigmas.
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

    PGOP_OUTER_LOOP(
        varying float sigma_j = sigmas[j];
        varying double max_bc = 0.0;
        for (uniform int32 m = 0; m < num_positions; ++m) {
            varying double bc = compute_Bhattacharyya_coefficient_gaussian(
                (varying double)pos_x[m], (varying double)pos_y[m], (varying double)pos_z[m],
                (varying double)sym_x, (varying double)sym_y, (varying double)sym_z,
                (varying double)sigma_j, (uniform double)sigmas[m]);
            max_bc = max(max_bc, bc);
        }
        varying double contribution = max_bc;
    )
}
