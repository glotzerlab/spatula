// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#if defined(__ARM_NEON)
#include "../data/RotationMatrix.h"
#include "../data/Vec3.h"
#include "../locality.h"
#include "../util/fastmath.h"
#include <algorithm>
#include <arm_neon.h>
#include <cmath>
#include <span>

namespace spatula { namespace computes {

/// Perform the subtraction of all fields and lanes of l, r.
inline float32x4x3_t neon_sub_3(float32x4x3_t l, float32x4x3_t r)
{
    return float32x4x3_t {vsubq_f32(l.val[0], r.val[0]),
                          vsubq_f32(l.val[1], r.val[1]),
                          vsubq_f32(l.val[2], r.val[2])};
}

/// Update a SIMD register storing the current minimum
inline float32x4_t neon_min_mag_sq(float32x4_t running_min, const float32x4x3_t x)
{
    float32x4_t mag_sq = vmulq_f32(x.val[0], x.val[0]); // (x**2)
    mag_sq = vfmaq_f32(mag_sq, x.val[1], x.val[1]);     // Accumulate y
    mag_sq = vfmaq_f32(mag_sq, x.val[2], x.val[2]);     // Accumulate z

    // Update the running minimum
    return vminq_f32(running_min, mag_sq);
}

inline float compute_pgop_gaussian_fast_neon(LocalNeighborhood& neighborhood,
                                             const std::span<const float> R_ij)
{
    const std::span<const data::Vec3> positions(neighborhood.rotated_positions);

    // Extract the raw underlying data, with some safety checks
    static_assert(sizeof(data::Vec3) == 3 * sizeof(float), "Vec3 must be tightly packed!");
    const float* raw_positions = reinterpret_cast<const float*>(positions.data());

    // NOTE: in src/PGOP.cc, we make the assumption that this function is only ever
    // called when all sigmas are constant. As such, we can precompute the denominator
    const double denom = 1.0 / (8.0 * neighborhood.sigmas[0] * neighborhood.sigmas[0]);
    double overlap = 0.0; // Accumulate in double, as the exp is done in f64
    data::RotationMatrix R;
    // loop over the R_ij. Each 3x3 segment is a symmetry operation
    // matrix. Each matrix should be applied to each point in positions.
    for (size_t i {0}; i < R_ij.size(); i += 9) {
        std::copy_n(R_ij.data() + i, 9, R.begin());

        // loop over positions
        for (size_t j {0}; j < positions.size(); j++) {
            const data::Vec3& p = positions[j];
            // symmetrized position is obtained by multiplying the operator with the position
            const data::Vec3 symmetrized_position = R.rotate(p);

            const auto symmetrized_pos_tiled = vld3q_dup_f32(&symmetrized_position.x);

            float32x4_t max_res = vdupq_n_f32(std::numeric_limits<float>::infinity());
            const size_t num_safe_simd = (positions.size() / 4) * 4;
            size_t m = 0;
            for (; m < num_safe_simd; m += 4) {
                auto pos_block = vld3q_f32(raw_positions + (m * 3));
                auto diff_block = neon_sub_3(pos_block, symmetrized_pos_tiled);

                // max(exp(-x)) == min(x)
                max_res = neon_min_mag_sq(max_res, diff_block);
            }

            // Collapse the 4 SIMD lanes to a single scalar minimum
            float final_min_dist = vminvq_f32(max_res);

            // Handle the remaining elements (<= 3)
            for (; m < positions.size(); ++m) {
                const data::Vec3& p = positions[m];
                const auto diff = p - symmetrized_position;

                // Update the final result
                final_min_dist = std::min(final_min_dist, diff.dot(diff));
            }

            // Final calculation using the collapsed minimum
            overlap += util::fast_exp_approx(-final_min_dist * denom);
        }
    }
    // cast to double to avoid integer division
    const auto normalization = static_cast<float>(positions.size() * R_ij.size()) / 9.0;
    return overlap / normalization;
}

}} // namespace spatula::computes

#endif // __ARM_NEON
