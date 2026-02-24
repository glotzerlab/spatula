// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86) \
    || defined(__aarch64__) || defined(_M_ARM64)
#include "../locality.h"
#include <span>

// Include the ISPC-generated headers
#include "pgop_fisher_ispc.h"
#include "pgop_gaussian_ispc.h"

namespace spatula { namespace computes {

/// Compute PGOP using ISPC (Intel SPMD Program Compiler) for SIMD.
inline float compute_pgop_gaussian_fast_ispc_wrapper(LocalNeighborhood& neighborhood,
                                                     const std::span<const float> R_ij)
{
    // NOTE: This function assumes all sigmas are constant (as with other fast variants)
    const float sigma = neighborhood.sigmas[0];
    const int32_t num_positions = static_cast<int32_t>(neighborhood.rotated_pos_x.size());
    const int32_t num_matrices = static_cast<int32_t>(R_ij.size() / 9);

    return ispc::compute_pgop_gaussian_fast_ispc(neighborhood.rotated_pos_x.data(),
                                                 neighborhood.rotated_pos_y.data(),
                                                 neighborhood.rotated_pos_z.data(),
                                                 R_ij.data(),
                                                 num_positions,
                                                 num_matrices,
                                                 sigma);
}

/// Compute PGOP Fisher.
inline float compute_pgop_fisher_fast_ispc_wrapper(LocalNeighborhood& neighborhood,
                                                   const std::span<const float> R_ij)
{
    const float kappa = neighborhood.sigmas[0];
    const int32_t num_positions = static_cast<int32_t>(neighborhood.rotated_pos_x.size());
    const int32_t num_matrices = static_cast<int32_t>(R_ij.size() / 9);

    return ispc::compute_pgop_fisher_fast_ispc(neighborhood.rotated_pos_x.data(),
                                               neighborhood.rotated_pos_y.data(),
                                               neighborhood.rotated_pos_z.data(),
                                               R_ij.data(),
                                               num_positions,
                                               num_matrices,
                                               kappa);
}

/// Compute PGOP Gaussian (non-fast, per-point sigmas).
inline float compute_pgop_gaussian_ispc_wrapper(LocalNeighborhood& neighborhood,
                                                const std::span<const float> R_ij)
{
    const int32_t num_positions = static_cast<int32_t>(neighborhood.rotated_pos_x.size());
    const int32_t num_matrices = static_cast<int32_t>(R_ij.size() / 9);

    return ispc::compute_pgop_gaussian_ispc(neighborhood.rotated_pos_x.data(),
                                            neighborhood.rotated_pos_y.data(),
                                            neighborhood.rotated_pos_z.data(),
                                            R_ij.data(),
                                            neighborhood.sigmas.data(),
                                            num_positions,
                                            num_matrices);
}

/// Compute PGOP Fisher (non-fast, per-point kappas).
inline float compute_pgop_fisher_ispc_wrapper(LocalNeighborhood& neighborhood,
                                              const std::span<const float> R_ij)
{
    const int32_t num_positions = static_cast<int32_t>(neighborhood.rotated_pos_x.size());
    const int32_t num_matrices = static_cast<int32_t>(R_ij.size() / 9);

    return ispc::compute_pgop_fisher_ispc(neighborhood.rotated_pos_x.data(),
                                          neighborhood.rotated_pos_y.data(),
                                          neighborhood.rotated_pos_z.data(),
                                          R_ij.data(),
                                          neighborhood.sigmas.data(),
                                          num_positions,
                                          num_matrices);
}

}} // namespace spatula::computes

#endif // x86/x64/ARM64 check
