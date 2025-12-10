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
let x = k * ln(2) + r // for some integer k and fractional remainder r.

// |r| must be < ln(2) / 2 as shown above
exp(x) = exp(k * ln(2) + r) = exp(k * ln(2)) * exp(r) = 2**k * exp(r)

// Rearrange to solve for k:
let k = round_toward_zero(x / ln(2)) = ⌊x / ln(2) + 1/2⌋

// Rearrange to solve for r:
let r = x - k * ln(2)
```

`2**k` has an exact binary representation and r is in [-ln(2)/2, ln(2)/2], meaning we've
reduced our unbounded range in a simple way! Note that, due to floating point math, this
approach still needs a bit of improviement. Cody and Waite proposed the following
approach:

https://arxiv.org/pdf/0708.3722v1
https://doc.lagout.org/science/0_Computer%20Science/2_Algorithms/Elementary%20Functions_%20Algorithms%20and%20Implementation%20%282nd%20ed.%29%20%5bMuller%202005-10-24%5d.pdf

// Solving for a correction factor (Remez coefficients)
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

For the Fisher distribution overlap, our expression is `sinh(x / 2) / x`. This is
challenging to approximate (1) because of numerical instability near zero and (2)
large values of x result in a large slope. As a result, we use the fixed limit of 0.5
near zero, std::sinh for 1e-6 < x < 24, and fast_exp_approx for large x. This guarantees
accuracy < 1e-7 for all regions of the curve, and still offers good performance.
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

inline double fast_exp_approx(double x)
{
    constexpr double ln2 = 0.69314718055994530941723;
    constexpr double ln2_recip = 1.44269504088896340;

    // Compute the float representation of k = ⌊x / ln(2) + 1/2⌋
    double k = std::round(x * ln2_recip);

    // FMA Cody-Waite Range reduction, skipping the low correction.
    double r = std::fma(-ln2, k, x);

    // Degree 5 Remez approximation from Sollya, using Estrin's method
    // p = (p0 := g + f * r) + r^2 * ((p1 := d + c * r) + r^2 * (p2 := b + a * r))
    constexpr double g = 1.0000000716546679769;
    constexpr double f = 0.99999969199097560324;
    constexpr double d = 0.4999889485139416001;
    constexpr double c = 0.16667574730852952047;
    constexpr double b = 4.191538198120380032e-2;
    constexpr double a = 8.2976549459683138915e-3;

    double r_sq = r * r;
    double p0 = std::fma(f, r, g);
    double p1 = std::fma(c, r, d);
    double p2 = std::fma(a, r, b);
    double p1_2 = std::fma(p2, r_sq, p1);
    double p = std::fma(p1_2, r_sq, p0);

    // Reconstruction: 2^k * p / (2x)
    uint64_t ki = static_cast<uint64_t>(k + 1023) << 52;
    double scale_factor = std::bit_cast<double, uint64_t>(ki);
    return p * scale_factor;
}

#if defined(__aarch64__) && !defined(SPATULA_DISABLE_NEON)
inline float64x2_t fast_exp_approx_simd(float64x2_t x)
{
    const float64x2_t neg_ln2 = vdupq_n_f64(-0.69314718055994530941723);
    const float64x2_t ln2_recip = vdupq_n_f64(1.44269504088896340);

    // Compute the float representation of k = ⌊x / ln(2) + 1/2⌋
    float64x2_t k = vrndnq_f64(vmulq_f64(x, ln2_recip));

    // FMA Cody-Waite Range reduction, skipping the low correction.
    float64x2_t r = vfmaq_f64(x, k, neg_ln2);

    // Degree 5 Remez approximation from Sollya, using Estrin's method
    // p = (p0 := g + f * r) + r^2 * ((p1 := d + c * r) + r^2 * (p2 := b + a * r))
    const float64x2_t g = vdupq_n_f64(1.0000000716546679769);
    const float64x2_t f = vdupq_n_f64(0.99999969199097560324);
    const float64x2_t d = vdupq_n_f64(0.4999889485139416001);
    const float64x2_t c = vdupq_n_f64(0.16667574730852952047);
    const float64x2_t b = vdupq_n_f64(4.191538198120380032e-2);
    const float64x2_t a = vdupq_n_f64(8.2976549459683138915e-3);

    // NOTE: vfmaq is a + (b*c), NOT (a * b) + c as std::fma
    float64x2_t r_sq = vmulq_f64(r, r);
    float64x2_t p0 = vfmaq_f64(g, r, f);
    float64x2_t p1 = vfmaq_f64(d, r, c);
    float64x2_t p2 = vfmaq_f64(b, r, a);
    float64x2_t p1_2 = vfmaq_f64(p1, r_sq, p2);
    float64x2_t p = vfmaq_f64(p0, r_sq, p1_2);

    // Reconstruction: 2^k * p / (2x) = (k as i64 << 52)
    int64x2_t k_shifted = vshlq_n_s64(vcvtq_s64_f64(k), 52);

    // Integer add to combine exponents, effectively computing p * 2^k
    return vreinterpretq_f64_s64(vaddq_s64(vreinterpretq_s64_f64(p), k_shifted));
}

#endif

}} // namespace spatula::util
