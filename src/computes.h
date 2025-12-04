// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once
#include "data/RotationMatrix.h"
#include "data/Vec3.h"
#include "locality.h"
#include "util/Metrics.h"
#include <algorithm>
#include <cmath>
#include <span>

namespace spatula { namespace computes {
double compute_pgop_gaussian(LocalNeighborhood& neighborhood, const std::span<const double> R_ij)
{
    const auto positions = neighborhood.rotated_positions;
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
            double max_res = 0.0;
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
    const auto normalization = static_cast<double>(positions.size() * R_ij.size()) / 9.0;
    return overlap / normalization;
}

double compute_pgop_gaussian_fast(LocalNeighborhood& neighborhood,
                                  const std::span<const double> R_ij)
{
    const auto positions = neighborhood.rotated_positions;
    // NOTE: in src/PGOP.cc, we make the assumption that this function is only ever
    // called when all sigmas are constant. As such, we can precompute the denominator
    const double denom = 1.0 / (8.0 * neighborhood.sigmas[0] * neighborhood.sigmas[0]);
    double overlap = 0.0;
    // loop over the R_ij. Each 3x3 segment is a symmetry operation
    // matrix. Each matrix should be applied to each point in positions.
    for (size_t i {0}; i < R_ij.size(); i += 9) {
        data::RotationMatrix R;
        std::copy_n(R_ij.data() + i, 9, R.begin());

        // loop over positions
        for (size_t j {0}; j < positions.size(); ++j) {
            const data::Vec3& p = positions[j];
            // symmetrized position is obtained by multiplying the operator with the position
            const data::Vec3 symmetrized_position = R.rotate(p);

            // compute overlap with every point in the positions
            double max_res = std::numeric_limits<double>::infinity();
            for (size_t m {0}; m < positions.size(); ++m) {
                data::Vec3 diff = positions[m] - symmetrized_position;
                // max(exp(-x)) == min(x)
                max_res = std::min(max_res, diff.dot(diff));
            }
            overlap += std::exp(-max_res * denom);
        }
    }
    // cast to double to avoid integer division
    const auto normalization = static_cast<double>(positions.size() * R_ij.size()) / 9.0;
    return overlap / normalization;
}
double compute_pgop_fisher(LocalNeighborhood& neighborhood, const std::span<const double> R_ij)
{
    const auto positions = neighborhood.rotated_positions;
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
    const auto normalization = static_cast<double>(positions.size() * R_ij.size()) / 9.0;
    return overlap / normalization;
}

inline double fast_sinhc_approx(double x)
{
    // Cody-Waite Constants
    constexpr double half_ln2_inv = 0.72134752044448170367996234050095; // 0.5 / ln(2)

    // High bits of ln(2), with trailing zeros for exact multiplication.
    constexpr double ln2_hi = 0.693147180369123816490;

    // "missing" low bits for ln2
    constexpr double ln2_lo = 1.90821492927058770002e-10;

    // Range Reduction: k_float = round(x/2 / ln2)
    double k_float = std::round(x * half_ln2_inv);
    int k = static_cast<int>(k_float);

    // r = (x/2) - k * ln2. We compute this as: r = 0.5 * x - k * ln2_hi - k * ln2_lo
    double r = 0.5 * x - k_float * ln2_hi;
    r -= k_float * ln2_lo;

    // Polynomial Approximation (Degree 5 Remez, should be within ~ 5e-7)
    constexpr double c5 = 1.0 / 120.0;
    constexpr double c4 = 1.0 / 24.0;
    constexpr double c3 = 1.0 / 6.0;
    constexpr double c2 = 0.5;

    // Evaluate (c4 + r*c5) and (c2 + r*c3) simultaneously (hopefully)
    double term_54 = c4 + r * c5;
    double term_32 = c2 + r * c3;

    // Evaluate the polynomial expansion
    double r_sq = r * r;
    double p = (1.0 + r) + r_sq * (term_32 + r_sq * term_54);

    // Reconstruction: 2^k * p / (2x), with a bias adjustment 1023 -> 1022 to halve x
    uint64_t ki = static_cast<uint64_t>(k + 1022) << 52;
    double scale_factor;
    std::memcpy(&scale_factor, &ki, sizeof(double));

    return p * (scale_factor / x);
}

double compute_pgop_fisher_fast(LocalNeighborhood& neighborhood, const std::span<const double> R_ij)
{
    const auto positions = neighborhood.rotated_positions;
    const double kappa = neighborhood.sigmas[0];
    const double prefix_term = 2.0 * kappa / std::sinh(kappa);
    double overlap = 0.0;
    // loop over the R_ij. Each 3x3 segment is a symmetry operation
    // matrix. Each matrix should be applied to each point in positions.
    data::RotationMatrix R;
    for (size_t i {0}; i < R_ij.size(); i += 9) {
        std::copy_n(R_ij.data() + i, 9, R.begin());
        // loop over positions
        for (size_t j {0}; j < positions.size(); ++j) {
            // symmetrized position is obtained by multiplying the operator with the position
            const data::Vec3 symmetrized_position = R.rotate(positions[j]);
            // Clamp lower bound to -1.0 in case our projection underflowed
            double max_proj = -1.0;
            for (size_t m {0}; m < positions.size(); ++m) {
                auto position = positions[m];
                double proj = position.dot(symmetrized_position);
                max_proj = std::max(proj, max_proj);
            }
            double inner_term = kappa * std::sqrt(2.0 * (1.0 + max_proj));

            if (inner_term > 16.0) { // error < 5e-7
                overlap += prefix_term * fast_sinhc_approx(inner_term);
                // overlap += prefix_term * std::sinh(inner_term * 0.5) / inner_term;
            } else if (inner_term > 1e-6) {
                // Use full-precision sinh to avoid errors when x is small
                overlap += prefix_term * std::sinh(inner_term * 0.5) / inner_term;
            } else {
                // Handle singularity at inner_term near 0 (when max_proj is near -1.0)
                overlap += prefix_term * 0.5;
            }
        }
    }
    // cast to double to avoid integer division
    const auto normalization = static_cast<double>(positions.size() * R_ij.size()) / 9.0;
    return overlap / normalization;
}
}} // namespace spatula::computes
