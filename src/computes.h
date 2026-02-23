// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once
#include "data/RotationMatrix.h"
#include "data/Vec3.h"
#include "locality.h"

// Include ISPC for SIMD (supports both x86 and ARM via NEON)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86) \
    || defined(__aarch64__) || defined(_M_ARM64)
#include "computes/ispc.h"
#define SPATULA_HAS_ISPC 1
#endif

// Keep native NEON as fallback for ARM
#if defined(__ARM_NEON) && !defined(SPATULA_HAS_ISPC)
#include "computes/neon.h"
#endif

#include "util/Metrics.h"
#include <algorithm>
#include <cmath>
#include <span>

namespace spatula { namespace computes {
float compute_pgop_gaussian(LocalNeighborhood& neighborhood, const std::span<const float> R_ij)
{
    const std::span<const data::Vec3> positions(neighborhood.rotated_positions);
    const auto sigmas = neighborhood.sigmas;
    double overlap = 0.0;
    // loop over the R_ij. Each 3x3 segment is a symmetry operation
    // matrix. Each matrix should be applied to each point in positions.
    for (size_t i {0}; i < R_ij.size(); i += 9) {
        data::RotationMatrix R;
        std::copy_n(R_ij.data() + i, 9, R.begin());

        // loop over positions
        for (size_t j {0}; j < positions.size(); ++j) {
            const auto& p = positions[j];
            // symmetrized position is obtained by multiplying the operator with the position
            const auto symmetrized_position = R.rotate(p);

            // compute overlap with every point in the positions
            float max_res = 0.0;
            for (size_t m {0}; m < positions.size(); ++m) {
                max_res = std::max(
                    max_res,
                    util::compute_Bhattacharyya_coefficient_gaussian(positions[m],
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

float compute_pgop_gaussian_fast(LocalNeighborhood& neighborhood, const std::span<const float> R_ij)
{
#if defined(SPATULA_HAS_ISPC)
    return compute_pgop_gaussian_fast_ispc_wrapper(neighborhood, R_ij);
#elif defined(__ARM_NEON)
    return compute_pgop_gaussian_fast_neon(neighborhood, R_ij);
#else
    const std::span<const data::Vec3> positions(neighborhood.rotated_positions);

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

            float max_res = std::numeric_limits<float>::infinity();
            for (size_t m {0}; m < positions.size(); ++m) {
                const data::Vec3& pos_m = positions[m];
                const auto diff = pos_m - symmetrized_position;

                // max(exp(-x)) == min(x)
                max_res = std::min(max_res, diff.dot(diff));
            }

            // Final calculation using the collapsed minimum
            overlap += util::fast_exp_approx(-max_res * denom);
        }
    }
    // cast to double to avoid integer division
    const auto normalization = static_cast<float>(positions.size() * R_ij.size()) / 9.0;
    return overlap / normalization;
#endif
}

float compute_pgop_fisher(LocalNeighborhood& neighborhood, const std::span<const float> R_ij)
{
    std::span<const data::Vec3> positions(neighborhood.rotated_positions);
    const auto sigmas = neighborhood.sigmas;
    double overlap = 0.0;
    // loop over the R_ij. Each 3x3 segment is a symmetry operation
    // matrix. Each matrix should be applied to each point in positions.
    for (size_t i {0}; i < R_ij.size(); i += 9) {
        data::RotationMatrix R;
        std::copy_n(R_ij.data() + i, 9, R.begin());
        // loop over positions
        for (size_t j {0}; j < positions.size(); ++j) {
            // symmetrized position is obtained by multiplying the operator with the position
            const data::Vec3 symmetrized_position = R.rotate(positions[j]);
            // compute overlap with every point in the positions
            double max_res = 0.0;
            for (size_t m {0}; m < positions.size(); ++m) {
                double BC = 0;
                BC = util::compute_Bhattacharyya_coefficient_fisher_normalized(positions[m],
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

float compute_pgop_fisher_fast(LocalNeighborhood& neighborhood, const std::span<const float> R_ij)
{
#if defined(SPATULA_HAS_ISPC)
    return compute_pgop_fisher_fast_ispc_wrapper(neighborhood, R_ij);
#else
    std::span<const data::Vec3> positions(neighborhood.rotated_positions);
    const double kappa = neighborhood.sigmas[0];
    const float prefix_term = 2.0 * kappa / std::sinh(kappa);
    float overlap = 0.0;
    data::RotationMatrix R;
    // loop over the R_ij. Each 3x3 segment is a symmetry operation
    // matrix. Each matrix should be applied to each point in positions.
    for (size_t i {0}; i < R_ij.size(); i += 9) {
        std::copy_n(R_ij.data() + i, 9, R.begin());

        for (size_t j {0}; j < positions.size(); ++j) {
            // symmetrized position is obtained by multiplying the operator with the position
            const data::Vec3 symmetrized_position = R.rotate(positions[j]);
            // Clamp lower bound to -1.0 in case our projection underflowed
            float max_proj = -1.0;
            for (size_t m {0}; m < positions.size(); ++m) {
                auto position = positions[m];
                float proj = position.dot(symmetrized_position);
                max_proj = std::max(proj, max_proj);
            }
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
#endif
}
}} // namespace spatula::computes
