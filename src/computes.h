// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once
#include "data/RotationMatrix.h"
#include "data/Vec3.h"
#include "locality.h"
#include "util/Metrics.h"
#include <algorithm>
#include <cmath>
#include <iostream>
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

inline double fast_exp_accurate(double x)
{
    // exp(x) = 2^k * exp(r)
    // k = round(x / ln2)
    const double ln2_inv = 1.44269504088896340735992468100189213;
    const double ln2_hi = 0.693147180559945309417232121458176568;
    const double ln2_lo = 1.9082149292705877e-10;

    double k_float = std::round(x * ln2_inv);
    int k = static_cast<int>(k_float);

    double r = x - k_float * ln2_hi;
    r -= k_float * ln2_lo; // Correction for higher precision

    // Polynomial Approximation for exp(r) on [-0.5, 0.5] (degree 4 taylor-remex)
    double p = 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;

    // return 2^k * p
    uint64_t ki = static_cast<uint64_t>(k + 1023) << 52;
    double two_to_k;
    // memcpy to avoid strict aliasing violation
    std::memcpy(&two_to_k, &ki, sizeof(double));

    return p * two_to_k;
}

inline double fast_sinhc_approx(double x)
{
    // sinh(x/2)*x approx exp(x/2) / (2x)
    return fast_exp_accurate(0.5 * x) / (2.0 * x);
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
            // Handle singularity at inner_term near 0 (when max_proj is near -1.0)
            // if (inner_term > 36.5) { // error < 1e-16
            if (inner_term > 1e-6) { // error < 1e-16
                // overlap += prefix_term * std::sinh(inner_term * 0.5) / inner_term;
                // overlap += prefix_term * std::exp(inner_term * 0.5) / (2.0 * inner_term);
                overlap += prefix_term * fast_sinhc_approx(inner_term);
                // } else if (inner_term > 1e-6) {
                //     overlap += prefix_term * std::sinh(inner_term * 0.5) / inner_term;
            } else {
                overlap += prefix_term * 0.5;
            }
        }
    }
    // cast to double to avoid integer division
    const auto normalization = static_cast<double>(positions.size() * R_ij.size()) / 9.0;
    return overlap / normalization;
}
}} // namespace spatula::computes
