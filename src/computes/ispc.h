// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86) \
    || defined(__aarch64__) || defined(_M_ARM64)
#include "../data/Vec3.h"
#include "../locality.h"
#include <span>
#include <vector>

// Include the ISPC-generated headers
#include "pgop_fisher_ispc.h"
#include "pgop_gaussian_ispc.h"

namespace spatula { namespace computes {

/// Compute PGOP using ISPC (Intel SPMD Program Compiler) for SIMD.
inline float compute_pgop_gaussian_fast_ispc_wrapper(LocalNeighborhood& neighborhood,
                                                     const std::span<const float> R_ij)
{
    const std::span<const data::Vec3> positions(neighborhood.rotated_positions);

    // Convert AoS (x0,y0,z0, x1,y1,z1, ...) to SoA (x[], y[], z[])
    // This enables ISPC to use contiguous vector loads instead of gathers
    const size_t n = positions.size();
    std::vector<float> pos_x(n), pos_y(n), pos_z(n);

    for (size_t i = 0; i < n; ++i) {
        pos_x[i] = positions[i].x;
        pos_y[i] = positions[i].y;
        pos_z[i] = positions[i].z;
    }

    // NOTE: This function assumes all sigmas are constant (as with other fast variants)
    const float sigma = neighborhood.sigmas[0];
    const int32_t num_positions = static_cast<int32_t>(n);
    const int32_t num_matrices = static_cast<int32_t>(R_ij.size() / 9);

    return ispc::compute_pgop_gaussian_fast_ispc(pos_x.data(),
                                                 pos_y.data(),
                                                 pos_z.data(),
                                                 R_ij.data(),
                                                 num_positions,
                                                 num_matrices,
                                                 sigma);
}

/// Compute PGOP Fisher using ISPC for SIMD.
inline float compute_pgop_fisher_fast_ispc_wrapper(LocalNeighborhood& neighborhood,
                                                   const std::span<const float> R_ij)
{
    const std::span<const data::Vec3> positions(neighborhood.rotated_positions);

    // Convert AoS (x0,y0,z0, x1,y1,z1, ...) to SoA (x[], y[], z[])
    const size_t n = positions.size();
    std::vector<float> pos_x(n), pos_y(n), pos_z(n);

    for (size_t i = 0; i < n; ++i) {
        pos_x[i] = positions[i].x;
        pos_y[i] = positions[i].y;
        pos_z[i] = positions[i].z;
    }

    const float kappa = neighborhood.sigmas[0];
    const int32_t num_positions = static_cast<int32_t>(n);
    const int32_t num_matrices = static_cast<int32_t>(R_ij.size() / 9);

    return ispc::compute_pgop_fisher_fast_ispc(pos_x.data(),
                                               pos_y.data(),
                                               pos_z.data(),
                                               R_ij.data(),
                                               num_positions,
                                               num_matrices,
                                               kappa);
}

}} // namespace spatula::computes

#endif // x86/x64/ARM64 check
