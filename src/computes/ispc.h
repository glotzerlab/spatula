// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86) \
    || defined(__aarch64__) || defined(_M_ARM64)
#include "../data/Vec3.h"
#include "../locality.h"
#include <span>

// Include the ISPC-generated header
#include "pgop_gaussian_ispc.h"

namespace spatula { namespace computes {

/// Compute PGOP using ISPC (Intel SPMD Program Compiler) for SIMD.
inline float compute_pgop_gaussian_fast_ispc_wrapper(LocalNeighborhood& neighborhood,
                                                     const std::span<const float> R_ij)
{
    const std::span<const data::Vec3> positions(neighborhood.rotated_positions);

    // Extract the raw underlying data, with safety check
    static_assert(sizeof(data::Vec3) == 3 * sizeof(float), "Vec3 must be tightly packed!");
    const float* raw_positions = reinterpret_cast<const float*>(positions.data());

    // NOTE: This function assumes all sigmas are constant (as with other fast variants)
    const float sigma = neighborhood.sigmas[0];
    const int32_t num_positions = static_cast<int32_t>(positions.size());
    const int32_t num_matrices = static_cast<int32_t>(R_ij.size() / 9);

    return ispc::compute_pgop_gaussian_fast_ispc(raw_positions,
                                                 R_ij.data(),
                                                 num_positions,
                                                 num_matrices,
                                                 sigma);
}

}} // namespace spatula::computes

#endif // x86/x64/ARM64 check
