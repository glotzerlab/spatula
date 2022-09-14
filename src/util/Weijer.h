#pragma once

#include <complex>
#include <vector>

namespace pgop { namespace util {
/**
 * @brief Given a WignerD matrix and spherical harmonic expansion coefficients \f$ Q_{m}^{l} \f$
 * compute the symmetrized expansion's coefficients.
 *
 * For reasons of performance this uses an existing vector's memory buffer to avoid memory
 * allocations.
 *
 * @param qlms The spherical harmonic expansion coefficients.
 * @param D_ij The WignerD matrix for a given symmetry or point group.
 * @param sym_qlm_buf The vector to place the symmetrized expansion coefficients into. For best
 * performance the capacity should be the size of qlms.
 * @param max_l The maximum \f$ l \f$ present in @p D_ij and @p qlms.
 */
void symmetrize_qlm(const std::vector<std::complex<double>>& qlms,
                    const std::vector<std::complex<double>>& D_ij,
                    std::vector<std::complex<double>>& sym_qlm_buf,
                    unsigned int max_l);
}} // namespace pgop::util
