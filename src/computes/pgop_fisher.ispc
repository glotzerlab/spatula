// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

// ISPC kernel for computing PGOP with Fisher (von Mises-Fisher) overlap.

#include "math.ispc"

// Fast version: assumes constant kappa.
export uniform float compute_pgop_fisher_fast_ispc(
    uniform const float pos_x[],
    uniform const float pos_y[],
    uniform const float pos_z[],
    uniform const float R_ij[],
    uniform int32 num_positions,
    uniform int32 num_matrices,
    uniform float kappa)
{
    varying double prefix_term = 2.0 * (uniform double)kappa / fast_sinh((uniform double)kappa);
    uniform double overlap = 0.0;

    PGOP_OUTER_LOOP(
        varying float max_proj = -1.0f;
        for (uniform int32 m = 0; m < num_positions; ++m) {
            varying float proj = dot_product(
                sym_x, sym_y, sym_z,
                pos_x[m], pos_y[m], pos_z[m]);
            max_proj = max(max_proj, proj);
        }

        varying double inner_term = (uniform double)kappa * sqrt(2.0 * (1.0 + (varying double)max_proj));
        varying double contribution;
        if (inner_term > 1e-6) {
            contribution = prefix_term * fast_sinh(inner_term * 0.5) / inner_term;
        } else {
            contribution = prefix_term * 0.5;
        }
    )
}

// Non-fast version: handles per-point kappas.
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

    PGOP_OUTER_LOOP(
        varying float kappa_j = kappas[j];
        varying double max_bc = 0.0;
        for (uniform int32 m = 0; m < num_positions; ++m) {
            varying double bc = compute_Bhattacharyya_coefficient_fisher_normalized(
                pos_x[m], pos_y[m], pos_z[m],
                sym_x, sym_y, sym_z,
                kappa_j, kappas[m]);
            max_bc = max(max_bc, bc);
        }
        varying double contribution = max_bc;
    )
}
