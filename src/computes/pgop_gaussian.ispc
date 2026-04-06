// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

// ISPC kernel for computing PGOP with Gaussian overlap.

#include "math.ispc"

#ifndef FLT_MAX
#define FLT_MAX 3.40282e+38f
#endif

// Fast version: assumes constant sigmas.
export uniform float compute_pgop_gaussian_fast_ispc(uniform const float pos_x[],
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
        varying float min_dist_sq = FLT_MAX;
        for (uniform int32 m = 0; m < num_positions; ++m) {
            varying float dist_sq
                = distance_squared(sym_x, sym_y, sym_z, pos_x[m], pos_y[m], pos_z[m]);
            min_dist_sq = min(min_dist_sq, dist_sq);
        }
        varying double contribution = fast_exp_approx(-min_dist_sq * denom);
    )
}

// Smooth version: replaces min with LogSumExp smooth upper bound.
// beta controls sharpness — larger beta approaches exact overlap.
export uniform float compute_pgop_gaussian_fast_smooth_ispc(uniform const float pos_x[],
                                                             uniform const float pos_y[],
                                                             uniform const float pos_z[],
                                                             uniform const float R_ij[],
                                                             uniform int32 num_positions,
                                                             uniform int32 num_matrices,
                                                             uniform float sigma,
                                                             uniform float beta)
{
    uniform double denom = 1.0 / (8.0 * (uniform double)sigma * (uniform double)sigma);
    uniform double overlap = 0.0;

    foreach_tiled(j = 0 ... num_positions)
    {
        varying float px = pos_x[j];
        varying float py = pos_y[j];
        varying float pz = pos_z[j];
        varying double position_total = 0.0;
        for (uniform int32 i = 0; i < num_matrices; ++i)
        {
            uniform RotationMatrix R = load_rotation_matrix(R_ij, i);
            varying float sym_x, sym_y, sym_z;
            apply_rotation(px, py, pz, R, sym_x, sym_y, sym_z);

            // First pass: find min squared distance for numerical stability.
            // Subtracting min_d from each term ensures exp(0) = 1 for the closest
            // point, preventing underflow in the LogSumExp sum.
            varying float min_dist_sq = FLT_MAX;
            for (uniform int32 m = 0; m < num_positions; ++m) {
                varying float dist_sq
                    = distance_squared(sym_x, sym_y, sym_z, pos_x[m], pos_y[m], pos_z[m]);
                min_dist_sq = min(min_dist_sq, dist_sq);
            }

            // Second pass: LogSumExp smooth-max of overlap exponents.
            // contribution = exp(-min_d*denom + (1/beta)*log(sum exp(-beta*(d_m-min_d)*denom)))
            // This gives a smooth upper bound on exp(-min_d*denom).
            varying double sum_exp = 0.0;
            for (uniform int32 m = 0; m < num_positions; ++m) {
                varying float dist_sq
                    = distance_squared(sym_x, sym_y, sym_z, pos_x[m], pos_y[m], pos_z[m]);
                sum_exp += fast_exp_approx(
                    -(double)(dist_sq - min_dist_sq) * (double)beta * denom);
            }

            varying double contribution = fast_exp_approx(
                -(double)min_dist_sq * denom + log(sum_exp) / (double)beta);
            position_total += contribution;
        }
        overlap += reduce_add(position_total);
    }
    return (uniform float)(overlap / (uniform double)(num_positions * num_matrices));
}

// Non-fast version: handles per-point sigmas.
export uniform float compute_pgop_gaussian_ispc(uniform const float pos_x[],
                                                uniform const float pos_y[],
                                                uniform const float pos_z[],
                                                uniform const float R_ij[],
                                                uniform const float sigmas[],
                                                uniform int32 num_positions,
                                                uniform int32 num_matrices)
{
    uniform double overlap = 0.0;

    PGOP_OUTER_LOOP(
        varying float sigma_j = sigmas[j]; varying double max_bc = 0.0;
        for (uniform int32 m = 0; m < num_positions; ++m) {
            varying double bc = compute_Bhattacharyya_coefficient_gaussian(pos_x[m],
                                                                           pos_y[m],
                                                                           pos_z[m],
                                                                           sym_x,
                                                                           sym_y,
                                                                           sym_z,
                                                                           sigma_j,
                                                                           sigmas[m]);
            max_bc = max(max_bc, bc);
        }
        varying double contribution = max_bc;
    )
}
