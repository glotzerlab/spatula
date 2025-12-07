// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

/* Fast, limited-precision implementations of elementary functions.

// Introduction
Common expressions like `exp`, `log`, and `sin/cos` are numerically implemented as
polynomial approximations. Standard library versions are typically accurate to 0-2 ULP,
requiring high degree approximations that in turn require a large number of
instructions. In SPATULA, our primary source of error stems from the optimization step
which can only guarantee a very rough lower bound of the actual OP value. As a result,
fast & limited-precision implementations of elementary functions can actually *improve*
the accuracy of PGOP by enabling more expensive optimization steps.

// Theory

// Approximations and Expansions
The most common polynomial approximation is a Taylor expansion: given a function f and n
derivatives at some point x, we can very accurately reproduce f(x) with a degree n
polynomial ḟ_n(x). However, the Taylor series minimizes the error only at x, meaning the
accuracy of our approximation is bad at nearly every other point.

An alternative approach uses Chebychev polynomials, which do a better job of spreading
out the error of our approximation for f(x). Chebychev approximations are well-studied
and frequently used, but it turns out we can do (slightly) better. By formulating our
problem as minimizing the L_∞ norm (maximum error) of f(x) - ḟ(x), we can apply the
Remez algorithm to find the optimal coefficients for ḟ(x). Interestingly, it turns out
that this approach is only optimal for real arithmetic: when calculating ḟ(x) in
floating point, naive rounding of the Remez coefficients can have unpredictably large
effects on the maximum error. Chevillard and Brisebarre proposed one final improvement,
a lattice reduction (or brute force optimization) step that further minimizes the L_∞
error over the set of floating point numbers.

[Talk on SLEEF](https://pdfs.semanticscholar.org/f993/776749e3a3e12836a1802b985cd7b524653d.pdf)
[Lattice Reduction](https://algo.inria.fr/seminars/sem06-07/chevillard-slides.pdf#page19)

// Range Reduction
It is important to note that polynomial approximations require a finite domain for
reasonable error convergence. It is trivial to construct some value x and finite n where
`exp(x)-ḟ_n(x)` has arbitrarily large absolute error. To solve this, our approximation
must also reduce the evaluation of the elementary function to some small range while
incurring as little error accumulation as possible. The following is standard for
`exp(x)`:

[Reference](https://justinwillmert.com/articles/2020/numerically-computing-the-exponential-function-with-polynomial-approximations/)

```
let x = (k + f) * ln(2) // for some integer k and fractional remainder f.
let r = ln(2) / f       // note that f began as a value in [-1/2, 1/2]

// |r| must be < ln(2) / 2 as shown above
exp(x) = exp(k * ln(2) + r) = exp(k * ln(2)) * exp(r) = 2**k * exp(r)

// Rearrange to solve for k:
let k = round_toward_zero(x / ln(2)) = ⌊x / ln(2) + 1/2⌋

// Rearrange to solve for r:
let r = x - k * ln(2)
```

`2**k` has an exact binary representation and r is in [-ln(2)/2, ln(2)/2], meaning we've
reduced our unbounded range in a simple, accurate way!

// Solving for Remez coefficients
The Sollya library is designed to solve for optimal coefficients for this sort of
polynomial approximation. The following script calculates the coefficients for a degree
5 Remez approximation of exp(x), with further optimization for floating point
coefficients. While this is not provably optimal, its far better than we really require.

```sollya
prec=64;
f=exp(x);
domain=[-log(2)/2;log(2)/2];

p = fpminimax(f, 5, [|D...|], domain);

maximum = dirtyinfnorm(p-f, domain);
print("The infinity norm of error is", maximum);
print("Optimized Polynomial:", p);
```

Output:
```
The infinity norm of error is 1.05976173233428312420268468938772131371726266388093e-7
Optimized Polynomial: 1.00000007165468307590572294429875910282135009765625 + x *
(0.99999969199162686006587819065316580235958099365234 + x *
(0.49998894851191177934879306121729314327239990234375 + x *
(0.166675747284489778055061037775885779410600662231445 + x *
(4.1915381994165779033778562734369188547134399414062e-2 + x
* 8.297655098188506939127506711884052492678165435791e-3))))
```

Note that, with degree 5, our error is below the floating point machine epsilon. This
boundary is fairly arbitrary but seems reasonable.

For the Fisher distribution overlap, our expression is `sinh(x / 2) / x`. Applying the
same approach as `exp(x)` shows degree 4 is required for an error below 1e-7. It turns
out we actually got lucky, with two coefficients being very small (< 1e-14). As a
result, our expression can be further approximated to the following while maintaining
near-optimal accuracy.

```
p = 0.5000000000838027425 + x * x * (2.0833320759058335941e-2 + x * x * 2.6069597217211469857e-4);
```
*/

#pragma once
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__aarch64__) && !defined(SPATULA_DISABLE_NEON)
#include <arm_neon.h>
#endif

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

    // Polynomial Approximation (Degree 5 Taylor, seems to be within ~ 5e-7)
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
// inline double fast_exp_approx(double x)
// {
//     // Cody-Waite Constants
//     constexpr double ln2_inv = 1.44269504088896340735992468100189; // 1 / ln(2)

//     // High bits of ln(2), with trailing zeros for exact multiplication.
//     constexpr double ln2_hi = 0.693147180369123816490;
//     constexpr double ln2_lo = 1.90821492927058770002e-10; // "missing" low bits for ln2

//     // Range Reduction: k_float = round(x / ln2)
//     double k_float = std::round(x * ln2_inv);
//     int k = static_cast<int>(k_float);

//     // r = x - k * ln2. We compute this as: r = x - k * ln2_hi - k * ln2_lo
//     double r = x - k_float * ln2_hi;
//     r -= k_float * ln2_lo;

//     // Polynomial Approximation (Degree 5 Remez, should be within ~ 5e-7)
//     constexpr double c5 = 1.0 / 120.0;
//     constexpr double c4 = 1.0 / 24.0;
//     constexpr double c3 = 1.0 / 6.0;
//     constexpr double c2 = 0.5;

//     // Evaluate (c4 + r*c5) and (c2 + r*c3) simultaneously (hopefully)
//     double term_54 = c4 + r * c5;
//     double term_32 = c2 + r * c3;

//     // Evaluate the polynomial expansion
//     double r_sq = r * r;
//     double p = (1.0 + r) + r_sq * (term_32 + r_sq * term_54);

//     // Reconstruction: 2^k * p / (2x), with a bias of 1023
//     uint64_t ki = static_cast<uint64_t>(k + 1023) << 52;
//     double scale_factor = std::bit_cast<double, uint64_t>(ki);

//     return p * scale_factor;
// }

#if defined(__aarch64__) && !defined(SPATULA_DISABLE_NEON)
inline float64x2_t fast_sinhc_approx_simd(float64x2_t x)
{
    // Cody-Waite Constants
    const float64x2_t half_ln2_inv = vdupq_n_f64(0.72134752044448170367996234050095); // 0.5 / ln(2)

    // High bits of ln(2), with trailing zeros for exact multiplication.
    const float64x2_t ln2_hi = vdupq_n_f64(0.693147180369123816490);

    // "missing" low bits for ln2
    const float64x2_t ln2_lo = vdupq_n_f64(1.90821492927058770002e-10);

    // Range Reduction: k_float = round(x/2 / ln2)
    float64x2_t k_float = vrndaq_f64(vmulq_f64(x, half_ln2_inv));

    // r = (x/2) - k * ln2. We compute this as: r = 0.5 * x - k * ln2_hi - k * ln2_lo
    float64x2_t r = vmlsq_f64(vmulq_f64(x, vdupq_n_f64(0.5)), k_float, ln2_hi);
    r = vmlsq_f64(r, k_float, ln2_lo);

    // Polynomial Approximation (Degree 5 taylor, should be within ~ 5e-7)
    const float64x2_t c5 = vdupq_n_f64(1.0 / 120.0);
    const float64x2_t c4 = vdupq_n_f64(1.0 / 24.0);
    const float64x2_t c3 = vdupq_n_f64(1.0 / 6.0);
    const float64x2_t c2 = vdupq_n_f64(0.5);

    // Evaluate (c4 + r*c5) and (c2 + r*c3) simultaneously (hopefully)
    float64x2_t term_54 = vfmaq_f64(c4, r, c5);
    float64x2_t term_32 = vfmaq_f64(c2, r, c3);

    // Evaluate the polynomial expansion
    float64x2_t r_sq = vmulq_f64(r, r);
    float64x2_t p = vfmaq_f64(term_32, r_sq, term_54);
    p = vmulq_f64(p, r_sq);
    p = vaddq_f64(p, vaddq_f64(vdupq_n_f64(1.0), r));

    // Reconstruction: 2^k * p / (2x), with a bias adjustment 1023 -> 1022 to halve x
    double k_float_arr[2];
    vst1q_f64(k_float_arr, k_float);
    int64_t k_arr[] = {static_cast<int64_t>(k_float_arr[0]), static_cast<int64_t>(k_float_arr[1])};

    uint64x2_t ki
        = {static_cast<uint64_t>(k_arr[0] + 1022), static_cast<uint64_t>(k_arr[1] + 1022)};
    ki = vshlq_n_u64(ki, 52);
    float64x2_t scale_factor = vreinterpretq_f64_u64(ki);

    return vmulq_f64(p, vdivq_f64(scale_factor, x));
}
inline float64x2_t fast_exp_approx_simd(float64x2_t x)
{
    // Cody-Waite Constants
    const float64x2_t ln2_inv = vdupq_n_f64(1.44269504088896340735992468100189); // 1 / ln(2)

    // High bits of ln(2), with trailing zeros for exact multiplication.
    const float64x2_t ln2_hi = vdupq_n_f64(0.693147180369123816490);
    const float64x2_t ln2_lo = vdupq_n_f64(1.90821492927058770002e-10);

    // Range Reduction: k_float = round(x / ln2)
    float64x2_t k_float = vrndaq_f64(vmulq_f64(x, ln2_inv));

    // r = x - k * ln2. We compute this as: r = x - k * ln2_hi - k * ln2_lo
    float64x2_t r = vmlsq_f64(x, k_float, ln2_hi);
    r = vmlsq_f64(r, k_float, ln2_lo);

    // Polynomial Approximation (Degree 5 taylor, should be within ~ 5e-7)
    const float64x2_t c5 = vdupq_n_f64(1.0 / 120.0);
    const float64x2_t c4 = vdupq_n_f64(1.0 / 24.0);
    const float64x2_t c3 = vdupq_n_f64(1.0 / 6.0);
    const float64x2_t c2 = vdupq_n_f64(0.5);

    // Evaluate (c4 + r*c5) and (c2 + r*c3)
    float64x2_t term_54 = vfmaq_f64(c4, r, c5);
    float64x2_t term_32 = vfmaq_f64(c2, r, c3);

    // Evaluate the polynomial expansion
    float64x2_t r_sq = vmulq_f64(r, r);
    float64x2_t p = vfmaq_f64(term_32, r_sq, term_54);
    p = vmulq_f64(p, r_sq);
    p = vaddq_f64(p, vaddq_f64(vdupq_n_f64(1.0), r));

    // Reconstruction: 2^k * p / (2x), with a bias of 1023
    int64x2_t k = vcvtq_s64_f64(k_float);

    // Add exponent bias
    k = vaddq_s64(k, vdupq_n_s64(1023));
    uint64x2_t ki = vshlq_n_u64(vreinterpretq_u64_s64(k), 52);

    // Reinterpret as double scale factors
    float64x2_t scale_factor = vreinterpretq_f64_u64(ki);

    return p * scale_factor;
}

#endif

}} // namespace spatula::util
