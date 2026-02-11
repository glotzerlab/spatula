// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once
#include "data/RotationMatrix.h"
#include "data/Vec3.h"
#include "locality.h"
#include <arm_neon.h>

#include "util/Metrics.h"
#include <algorithm>
#include <cmath>
#include <span>

namespace spatula { namespace computes {
float compute_pgop_gaussian(LocalNeighborhood<float>& neighborhood,
                            const std::span<const float> R_ij)
{
    const std::span<const data::Vec3<float>> positions(neighborhood.rotated_positions);
    const auto sigmas = neighborhood.sigmas;
    double overlap = 0.0;
    // loop over the R_ij. Each 3x3 segment is a symmetry operation
    // matrix. Each matrix should be applied to each point in positions.
    for (size_t i {0}; i < R_ij.size(); i += 9) {
        data::RotationMatrix<float> R;
        std::copy_n(R_ij.data() + i, 9, R.begin());

        // loop over positions
        for (size_t j {0}; j < positions.size(); ++j) {
            // Load and deinterleave 4 Vec3f into a float32x4x3_t struct.
            const auto& p = positions[j];
            // symmetrized position is obtained by multiplying the operator with the position
            const auto symmetrized_position = R.rotate(p);

            // compute overlap with every point in the positions
            float max_res = 0.0;
            for (size_t m {0}; m < positions.size(); ++m) {
                max_res = std::max(
                    max_res,
                    util::compute_Bhattacharyya_coefficient_gaussian<float>(positions[m],
                                                                            symmetrized_position,
                                                                            sigmas[j],
                                                                            sigmas[m]));
            }
            overlap += max_res;
        }
    }
    // cast to double to avoid integer division
    const auto normalization = static_cast<float>(positions.size() * R_ij.size()) / 9.0;
    return overlap / normalization;
}

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
float compute_pgop_gaussian_fast(LocalNeighborhood<float>& neighborhood,
                                 const std::span<const float> R_ij)
{
    const std::span<const data::Vec3<float>> positions(neighborhood.rotated_positions);

    // Extract the raw underlying data, with some safety checks
    static_assert(sizeof(data::Vec3<float>) == 3 * sizeof(float), "Vec3 must be tightly packed!");
    const float* raw_positions = reinterpret_cast<const float*>(positions.data());

    // NOTE: in src/PGOP.cc, we make the assumption that this function is only ever
    // called when all sigmas are constant. As such, we can precompute the denominator
    const double denom = 1.0 / (8.0 * neighborhood.sigmas[0] * neighborhood.sigmas[0]);
    double overlap = 0.0; // Accumulate in double, as the exp is done in f64
    data::RotationMatrix<float> R;
    // loop over the R_ij. Each 3x3 segment is a symmetry operation
    // matrix. Each matrix should be applied to each point in positions.
    for (size_t i {0}; i < R_ij.size(); i += 9) {
        std::copy_n(R_ij.data() + i, 9, R.begin());

        // loop over positions
        for (size_t j {0}; j < positions.size(); j++) {
            const data::Vec3<float>& p = positions[j];
            // symmetrized position is obtained by multiplying the operator with the position
            const data::Vec3<float> symmetrized_position = R.rotate(p);

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
                const data::Vec3<float>& p = positions[m];
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
float compute_pgop_fisher(LocalNeighborhood<float>& neighborhood, const std::span<const float> R_ij)
{
    std::span<const data::Vec3<float>> positions(neighborhood.rotated_positions);
    const auto sigmas = neighborhood.sigmas;
    double overlap = 0.0;
    // loop over the R_ij. Each 3x3 segment is a symmetry operation
    // matrix. Each matrix should be applied to each point in positions.
    for (size_t i {0}; i < R_ij.size(); i += 9) {
        data::RotationMatrix<float> R;
        std::copy_n(R_ij.data() + i, 9, R.begin());
        // loop over positions
        for (size_t j {0}; j < positions.size(); ++j) {
            // symmetrized position is obtained by multiplying the operator with the position
            const data::Vec3<float> symmetrized_position = R.rotate(positions[j]);
            // compute overlap with every point in the positions
            double max_res = 0.0;
            for (size_t m {0}; m < positions.size(); ++m) {
                double BC = 0;
                BC = util::compute_Bhattacharyya_coefficient_fisher_normalized<float>(
                    positions[m],
                    symmetrized_position,
                    sigmas[j],
                    sigmas[m]);
                if (BC > max_res)
                    max_res = BC;
            }
            overlap += max_res;
        }
    }
    // cast to double to avoid integer division
    const float normalization = static_cast<float>(positions.size() * R_ij.size()) / 9.0;
    return overlap / normalization;
}

float compute_pgop_fisher_fast(LocalNeighborhood<float>& neighborhood,
                               const std::span<const float> R_ij)

{
    std::span<const data::Vec3<float>> positions(neighborhood.rotated_positions);
    const double kappa = neighborhood.sigmas[0];
    const float prefix_term = 2.0 * kappa / std::sinh(kappa);
    float overlap = 0.0;
    data::RotationMatrix<float> R;
    // loop over the R_ij. Each 3x3 segment is a symmetry operation
    // matrix. Each matrix should be applied to each point in positions.
    for (size_t i {0}; i < R_ij.size(); i += 9) {
        std::copy_n(R_ij.data() + i, 9, R.begin());

        for (size_t j {0}; j < positions.size(); ++j) {
            // symmetrized position is obtained by multiplying the operator with the position
            const data::Vec3<float> symmetrized_position = R.rotate(positions[j]);
            // Clamp lower bound to -1.0 in case our projection underflowed
            float max_proj = -1.0;
            for (size_t m {0}; m < positions.size(); ++m) {
                auto position = positions[m];
                float proj = position.dot(symmetrized_position);
                max_proj = std::max(proj, max_proj);
            }
            // floatODO: fast sinhf
            double inner_term = kappa * std::sqrt(2.0 * (1.0 + max_proj));
            if (inner_term > 1e-6) {
                // Use double-precision sinh to avoid errors when x is small
                overlap += prefix_term * std::sinh(inner_term * 0.5) / inner_term;
            } else {
                // Handle singularity at inner_term near 0 (when max_proj is near -1.0)
                overlap += prefix_term * 0.5;
            }
        }
    }

    // cast to double to avoid integer division
    const float normalization = static_cast<float>(positions.size() * R_ij.size()) / 9.0;

    return overlap / normalization;
}
}} // namespace spatula::computes
