// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

namespace spatula { namespace util {
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
    double scale_factor = std::bit_cast<double, uint64_t>(ki);

    return p * (scale_factor / x);
}

}} // namespace spatula::util
