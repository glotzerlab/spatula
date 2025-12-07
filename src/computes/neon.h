// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once
#include "data/RotationMatrix.h"
#include "data/Vec3.h"
#include "locality.h"
#if defined(__aarch64__) && !defined(SPATULA_DISABLE_NEON)
#include <arm_neon.h>
#endif
#include "util/Metrics.h"
#include <algorithm>
#include <cmath>
#include <span>

namespace spatula { namespace computes {

double compute_pgop_gaussian_fast_neon(LocalNeighborhood& neighborhood,
                                       const std::span<const double> R_ij)
{
    std::span<const data::Vec3> positions(neighborhood.rotated_positions);
    // NOTE: in src/PGOP.cc, we make the assumption that this function is only ever
    // called when all sigmas are constant. As such, we can precompute the denominator
    const double denom = 1.0 / (8.0 * neighborhood.sigmas[0] * neighborhood.sigmas[0]);
    double overlap = 0.0;
    data::RotationMatrix R;
    // loop over the R_ij. Each 3x3 segment is a symmetry operation
    // matrix. Each matrix should be applied to each point in positions.
    for (size_t i {0}; i < R_ij.size(); i += 9) {
        std::copy_n(R_ij.data() + i, 9, R.begin());

        // loop over positions
        size_t j = 0;
        for (; j + 1 < positions.size(); j += 2) {
            // symmetrized position is obtained by multiplying the operator with the position
            const data::Vec3 s_p_0 = R.rotate(positions[j]);
            const data::Vec3 s_p_1 = R.rotate(positions[j + 1]);
            // compute overlap with every point in the positions
            float64x2_t min_dist_sq_vec = vdupq_n_f64(std::numeric_limits<double>::infinity());
            const float64x2_t s_x = {s_p_0.x, s_p_1.x};
            const float64x2_t s_y = {s_p_0.y, s_p_1.y};
            const float64x2_t s_z = {s_p_0.z, s_p_1.z};
            for (size_t m = 0; m < positions.size(); ++m) {
                const float64x2_t p_x = vdupq_n_f64(positions[m].x);
                const float64x2_t p_y = vdupq_n_f64(positions[m].y);
                const float64x2_t p_z = vdupq_n_f64(positions[m].z);
                float64x2_t diff_x = vsubq_f64(p_x, s_x);
                float64x2_t diff_y = vsubq_f64(p_y, s_y);
                float64x2_t diff_z = vsubq_f64(p_z, s_z);

                float64x2_t sq_dist_vec = vmulq_f64(diff_x, diff_x);
                sq_dist_vec = vfmaq_f64(sq_dist_vec, diff_y, diff_y);
                sq_dist_vec = vfmaq_f64(sq_dist_vec, diff_z, diff_z);
                min_dist_sq_vec = vminq_f64(min_dist_sq_vec, sq_dist_vec);
            }

            float64x2_t overlaps
                = util::fast_exp_approx_simd(vnegq_f64(min_dist_sq_vec * vdupq_n_f64(denom)));
            // Horizontal add
            overlap += vaddvq_f64(overlaps);
        }

        for (; j < positions.size(); ++j) {
            const data::Vec3 symmetrized_position = R.rotate(positions[j]);

            // compute overlap with every point in the positions
            double max_res = std::numeric_limits<double>::infinity();
            for (size_t m {0}; m < positions.size(); ++m) {
                data::Vec3 diff = positions[m] - symmetrized_position;
                // max(exp(-x)) == min(x)
                max_res = std::min(max_res, diff.dot(diff));
            }
            overlap += util::fast_exp_approx(-max_res * denom);
        }
    }
    // cast to double to avoid integer division
    const auto normalization = static_cast<double>(positions.size() * R_ij.size()) / 9.0;
    return overlap / normalization;
}

double compute_pgop_fisher_fast_neon(LocalNeighborhood& neighborhood,
                                     const std::span<const double> R_ij)

{
    std::span<const data::Vec3> positions(neighborhood.rotated_positions);
    const double kappa = neighborhood.sigmas[0];

    const float64x2_t fast_path_lower_bound = vdupq_n_f64(24.0);
    const float64x2_t k = vdupq_n_f64(kappa);
    const float64x2_t half = vdupq_n_f64(0.5);
    const float64x2_t one = vdupq_n_f64(1.0);
    const float64x2_t two = vdupq_n_f64(2.0);

    const double prefix_term = 2.0 * kappa / std::sinh(kappa);
    double overlap = 0.0;
    data::RotationMatrix R;
    // loop over the R_ij. Each 3x3 segment is a symmetry operation
    // matrix. Each matrix should be applied to each point in positions.
    for (size_t i {0}; i < R_ij.size(); i += 9) {
        std::copy_n(R_ij.data() + i, 9, R.begin());

        // loop over positions
        size_t j = 0;
        for (; j + 1 < positions.size(); j += 2) {
            // symmetrized position is obtained by multiplying the operator with the position

            const data::Vec3 s_p_0 = R.rotate(positions[j]);
            const data::Vec3 s_p_1 = R.rotate(positions[j + 1]);
            // compute overlap with every point in the positions
            float64x2_t max_proj_vec = vdupq_n_f64(-1.0);
            const float64x2_t s_x = {s_p_0.x, s_p_1.x};
            const float64x2_t s_y = {s_p_0.y, s_p_1.y};
            const float64x2_t s_z = {s_p_0.z, s_p_1.z};
            for (size_t m = 0; m < positions.size(); ++m) {
                const float64x2_t p_x = vdupq_n_f64(positions[m].x);
                const float64x2_t p_y = vdupq_n_f64(positions[m].y);
                const float64x2_t p_z = vdupq_n_f64(positions[m].z);
                float64x2_t proj_vec = vmulq_f64(p_x, s_x);
                proj_vec = vfmaq_f64(proj_vec, p_y, s_y);
                proj_vec = vfmaq_f64(proj_vec, p_z, s_z);
                max_proj_vec = vmaxq_f64(max_proj_vec, proj_vec);
            }

            float64x2_t inner_term
                = vmulq_f64(k, vsqrtq_f64(vmulq_f64(two, vaddq_f64(one, max_proj_vec))));

            // Create a mask where bits are 1 if (> 24.0), 0 otherwise.
            uint64x2_t mask = vcgtq_f64(inner_term, fast_path_lower_bound);

            // Check if all lanes are small enough to take the `exp` path.
            // vminvq_u32 returns 0xFFFFFFFF (-1) only if every bit in the vector is 1.
            if (vminvq_u32(vreinterpretq_u32_u64(mask)) == 0xFFFFFFFF) {
                float64x2_t denom = vmulq_f64(inner_term, two);
                float64x2_t res = util::fast_exp_approx_simd(vmulq_f64(inner_term, half));
                res = vdivq_f64(res, denom);

                // Horizontal add
                overlap += prefix_term * vaddvq_f64(res);

            } else {
                double inner_arr[2];
                vst1q_f64(inner_arr, inner_term);

                // Unroll loop for the two lanes
                if (inner_arr[0] > 1e-6) {
                    overlap += prefix_term * std::sinh(inner_arr[0] * 0.5) / inner_arr[0];
                } else {
                    overlap += prefix_term * 0.5;
                }
                if (inner_arr[1] > 1e-6) {
                    overlap += prefix_term * std::sinh(inner_arr[1] * 0.5) / inner_arr[1];
                } else {
                    overlap += prefix_term * 0.5;
                }
            }
        }

        for (; j < positions.size(); ++j) {
            const data::Vec3 symmetrized_position = R.rotate(positions[j]);
            double max_proj = -1.0;
            for (size_t m {0}; m < positions.size(); ++m) {
                max_proj = std::max(max_proj, positions[m].dot(symmetrized_position));
            }
            double inner_term = kappa * std::sqrt(2.0 * (1.0 + max_proj));

            if (inner_term > 1e-6) {
                overlap += prefix_term * std::sinh(inner_term * 0.5) / inner_term;
            } else {
                overlap += prefix_term * 0.5;
            }
        }
    }

    // cast to double to avoid integer division
    const double normalization = static_cast<double>(positions.size() * R_ij.size()) / 9.0;

    return overlap / normalization;
}

}} // namespace spatula::computes
