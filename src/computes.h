// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once
#include "data/RotationMatrix.h"
#include "data/Vec3.h"
#include "locality.h"

// Include ISPC for SIMD (controlled by ENABLE_ISPC CMake option)
#if defined(SPATULA_HAS_ISPC)
#include "computes/ispc.h"
#endif

#include "util/Metrics.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>

namespace spatula { namespace computes {
float compute_pgop_gaussian(LocalNeighborhood& neighborhood, const std::span<const float> R_ij)
{
#if defined(SPATULA_HAS_ISPC)
    return compute_pgop_gaussian_ispc_wrapper(neighborhood, R_ij);
#else
    const float* pos_x = neighborhood.rotated_pos_x.data();
    const float* pos_y = neighborhood.rotated_pos_y.data();
    const float* pos_z = neighborhood.rotated_pos_z.data();
    const size_t n = neighborhood.rotated_pos_x.size();

    const auto sigmas = neighborhood.sigmas;
    double overlap = 0.0;
    // loop over the R_ij. Each 3x3 segment is a symmetry operation
    // matrix. Each matrix should be applied to each point in positions.
    for (size_t i {0}; i < R_ij.size(); i += 9) {
        data::RotationMatrix R;
        std::copy_n(R_ij.data() + i, 9, R.begin());

        // loop over positions
        for (size_t j {0}; j < n; ++j) {
            // symmetrized position is obtained by multiplying the operator with the position
            data::Vec3 symmetrized_position {R[0] * pos_x[j] + R[1] * pos_y[j] + R[2] * pos_z[j],
                                             R[3] * pos_x[j] + R[4] * pos_y[j] + R[5] * pos_z[j],
                                             R[6] * pos_x[j] + R[7] * pos_y[j] + R[8] * pos_z[j]};

            // compute overlap with every point in the positions
            float max_res = 0.0;
            for (size_t m {0}; m < n; ++m) {
                max_res = std::max(max_res,
                                   util::compute_Bhattacharyya_coefficient_gaussian(
                                       data::Vec3 {pos_x[m], pos_y[m], pos_z[m]},
                                       symmetrized_position,
                                       sigmas[j],
                                       sigmas[m]));
            }
            overlap += max_res;
        }
    }
    // cast to double to avoid integer division
    const auto normalization = static_cast<float>(n * R_ij.size()) / 9.0;
    return overlap / normalization;
#endif
}

float compute_pgop_gaussian_fast(LocalNeighborhood& neighborhood, const std::span<const float> R_ij)
{
#if defined(SPATULA_HAS_ISPC)
    return compute_pgop_gaussian_fast_ispc_wrapper(neighborhood, R_ij);
#else
    const float* pos_x = neighborhood.rotated_pos_x.data();
    const float* pos_y = neighborhood.rotated_pos_y.data();
    const float* pos_z = neighborhood.rotated_pos_z.data();
    const size_t n = neighborhood.rotated_pos_x.size();

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
        for (size_t j {0}; j < n; j++) {
            // symmetrized position is obtained by multiplying the operator with the position
            const float sym_x = R[0] * pos_x[j] + R[1] * pos_y[j] + R[2] * pos_z[j];
            const float sym_y = R[3] * pos_x[j] + R[4] * pos_y[j] + R[5] * pos_z[j];
            const float sym_z = R[6] * pos_x[j] + R[7] * pos_y[j] + R[8] * pos_z[j];

            float min_dist_sq = std::numeric_limits<float>::infinity();
            for (size_t m {0}; m < n; ++m) {
                const float dx = pos_x[m] - sym_x;
                const float dy = pos_y[m] - sym_y;
                const float dz = pos_z[m] - sym_z;

                // max(exp(-x)) == min(x)
                min_dist_sq = std::min(min_dist_sq, dx * dx + dy * dy + dz * dz);
            }

            // Final calculation using the collapsed minimum
            overlap += util::fast_exp_approx(-min_dist_sq * denom);
        }
    }
    // cast to double to avoid integer division
    const auto normalization = static_cast<float>(n * R_ij.size()) / 9.0;
    return overlap / normalization;
#endif
}

float compute_pgop_gaussian_fast_smooth(LocalNeighborhood& neighborhood,
                                        const std::span<const float> R_ij,
                                        float beta)
{
    const float* pos_x = neighborhood.rotated_pos_x.data();
    const float* pos_y = neighborhood.rotated_pos_y.data();
    const float* pos_z = neighborhood.rotated_pos_z.data();
    const size_t n = neighborhood.rotated_pos_x.size();

    const double sigma = static_cast<double>(neighborhood.sigmas[0]);
    const double denom = 1.0 / (8.0 * sigma * sigma);
    const double beta_d = static_cast<double>(beta);
    // Precompute the LogSumExp weight scale: beta * denom.
    // Clamp to prevent exp(-weight_scale * delta) from underflowing to 0
    // for all terms (which would make log(sum) = -inf).
    const double weight_scale = std::min(beta_d * denom, 700.0);
    double overlap = 0.0;
    data::RotationMatrix R;
    // loop over the R_ij. Each 3x3 segment is a symmetry operation
    // matrix. Each matrix should be applied to each point in positions.
    for (size_t i {0}; i < R_ij.size(); i += 9) {
        std::copy_n(R_ij.data() + i, 9, R.begin());

        // loop over positions
        for (size_t j {0}; j < n; j++) {
            // symmetrized position — promote to double for precision
            const double sym_x = static_cast<double>(R[0]) * pos_x[j]
                                + static_cast<double>(R[1]) * pos_y[j]
                                + static_cast<double>(R[2]) * pos_z[j];
            const double sym_y = static_cast<double>(R[3]) * pos_x[j]
                                + static_cast<double>(R[4]) * pos_y[j]
                                + static_cast<double>(R[5]) * pos_z[j];
            const double sym_z = static_cast<double>(R[6]) * pos_x[j]
                                + static_cast<double>(R[7]) * pos_y[j]
                                + static_cast<double>(R[8]) * pos_z[j];

            // First pass: find min squared distance (in double precision).
            double min_dist_sq = std::numeric_limits<double>::infinity();
            for (size_t m {0}; m < n; ++m) {
                const double dx = static_cast<double>(pos_x[m]) - sym_x;
                const double dy = static_cast<double>(pos_y[m]) - sym_y;
                const double dz = static_cast<double>(pos_z[m]) - sym_z;
                const double d_sq = dx * dx + dy * dy + dz * dz;
                if (d_sq < min_dist_sq) min_dist_sq = d_sq;
            }

            // Second pass: LogSumExp smooth-max of overlap exponents.
            // contribution = exp(-min_d*denom + (1/beta)*log(sum exp(-beta*(d_m-min_d)*denom)))
            // This gives a smooth upper bound on the exact exp(-min_d*denom).
            // Using std::exp for full double-precision correctness (needed for AD).
            double sum_exp = 0.0;
            for (size_t m {0}; m < n; ++m) {
                const double dx = static_cast<double>(pos_x[m]) - sym_x;
                const double dy = static_cast<double>(pos_y[m]) - sym_y;
                const double dz = static_cast<double>(pos_z[m]) - sym_z;
                const double d_sq = dx * dx + dy * dy + dz * dz;
                sum_exp += std::exp(-(d_sq - min_dist_sq) * weight_scale);
            }

            const double exp_arg = -min_dist_sq * denom + std::log(sum_exp) / beta_d;
            overlap += std::exp(std::min(exp_arg, 700.0));
        }
    }
    const double normalization = static_cast<double>(n * R_ij.size()) / 9.0;
    const double result = overlap / normalization;
    // Guard against NaN (from 0/0) or overflow when casting to float.
    // The smooth LogSumExp upper bound is at most n^(1/beta), typically < 2.0.
    if (!std::isfinite(result) || result < 0.0) {
        return 0.0f;
    }
    return static_cast<float>(std::min(result, 2.0));
}

float compute_pgop_fisher(LocalNeighborhood& neighborhood, const std::span<const float> R_ij)
{
#if defined(SPATULA_HAS_ISPC)
    return compute_pgop_fisher_ispc_wrapper(neighborhood, R_ij);
#else
    const float* pos_x = neighborhood.rotated_pos_x.data();
    const float* pos_y = neighborhood.rotated_pos_y.data();
    const float* pos_z = neighborhood.rotated_pos_z.data();
    const size_t n = neighborhood.rotated_pos_x.size();

    const auto sigmas = neighborhood.sigmas;
    double overlap = 0.0;
    // loop over the R_ij. Each 3x3 segment is a symmetry operation
    // matrix. Each matrix should be applied to each point in positions.
    for (size_t i {0}; i < R_ij.size(); i += 9) {
        data::RotationMatrix R;
        std::copy_n(R_ij.data() + i, 9, R.begin());
        // loop over positions
        for (size_t j {0}; j < n; ++j) {
            // symmetrized position is obtained by multiplying the operator with the position
            data::Vec3 symmetrized_position {R[0] * pos_x[j] + R[1] * pos_y[j] + R[2] * pos_z[j],
                                             R[3] * pos_x[j] + R[4] * pos_y[j] + R[5] * pos_z[j],
                                             R[6] * pos_x[j] + R[7] * pos_y[j] + R[8] * pos_z[j]};
            // compute overlap with every point in the positions
            double max_res = 0.0;
            for (size_t m {0}; m < n; ++m) {
                double BC = 0;
                BC = util::compute_Bhattacharyya_coefficient_fisher_normalized(
                    data::Vec3 {pos_x[m], pos_y[m], pos_z[m]},
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
    const float normalization = static_cast<float>(n * R_ij.size()) / 9.0;
    return overlap / normalization;
#endif
}

float compute_pgop_fisher_fast(LocalNeighborhood& neighborhood, const std::span<const float> R_ij)
{
#if defined(SPATULA_HAS_ISPC)
    return compute_pgop_fisher_fast_ispc_wrapper(neighborhood, R_ij);
#else
    const float* pos_x = neighborhood.rotated_pos_x.data();
    const float* pos_y = neighborhood.rotated_pos_y.data();
    const float* pos_z = neighborhood.rotated_pos_z.data();
    const size_t n = neighborhood.rotated_pos_x.size();

    const double kappa = neighborhood.sigmas[0];
    const float prefix_term = 2.0 * kappa / std::sinh(kappa);
    float overlap = 0.0;
    data::RotationMatrix R;
    // loop over the R_ij. Each 3x3 segment is a symmetry operation
    // matrix. Each matrix should be applied to each point in positions.
    for (size_t i {0}; i < R_ij.size(); i += 9) {
        std::copy_n(R_ij.data() + i, 9, R.begin());

        for (size_t j {0}; j < n; ++j) {
            // symmetrized position is obtained by multiplying the operator with the position
            const float sym_x = R[0] * pos_x[j] + R[1] * pos_y[j] + R[2] * pos_z[j];
            const float sym_y = R[3] * pos_x[j] + R[4] * pos_y[j] + R[5] * pos_z[j];
            const float sym_z = R[6] * pos_x[j] + R[7] * pos_y[j] + R[8] * pos_z[j];

            // Clamp lower bound to -1.0 in case our projection underflowed
            float max_proj = -1.0;
            for (size_t m {0}; m < n; ++m) {
                float proj = pos_x[m] * sym_x + pos_y[m] * sym_y + pos_z[m] * sym_z;
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
    const float normalization = static_cast<float>(n * R_ij.size()) / 9.0;

    return overlap / normalization;
#endif
}
}} // namespace spatula::computes
