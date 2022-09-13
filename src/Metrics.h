#pragma once

#include <complex>
#include <vector>

/**
 * @brief compute the Pearson correlation between \f$ B \f$ and \f$ B_{sym, i}, \forall i \f$ where
 * \f$ B \f$ is the bond order diagram and \f$ i \f$ is the point group. We do this through the
 * spherical harmonic expansion of \f$ B \f$ and its symmetrized expansions through the coefficients
 * \f$ Q_{m}^{l} \f$.
 *
 * The implementation uses some tricks to make the computation as efficient as possible compared to
 * a standard corrlation computation.
 *
 * @param qlms The coefficents for the spherical harmonic expansion of \f$ B \f$.
 * @param sym_qlms The coefficents for all symmetrized spherical harmonic expansion of
 * \f$ B_{sym, i} \f$ where \f$ i \f$ indicates which point group symmetrization has been done.
 * @returns A vector of the Pearson correlation for each point group symmetrization.
 */
std::vector<double> covariance(const std::vector<std::complex<double>>& qlms,
                               const std::vector<std::vector<std::complex<double>>>& sym_qlms);

/**
 * @brief compute the Pearson correlation between \f$ B \f$ and \f$ B_{sym}  \f$ where \f$ B \f$ is
 * the bond order diagram. We do this through the spherical harmonic expansion of \f$ B \f$ and its
 * symmetrized expansions through the coefficients \f$ Q_{m}^{l} \f$.
 *
 * The implementation uses some tricks to make the computation as efficient as possible compared to
 * a standard corrlation computation.
 *
 * @param qlms The coefficents for the spherical harmonic expansion of \f$ B \f$.
 * @param sym_qlms The coefficents for the symmetrized spherical harmonic expansion of
 * \f$ B_{sym} \f$.
 * @returns A vector of the Pearson correlation for the point group symmetrization.
 */
double covariance(const std::vector<std::complex<double>>& qlms,
                  const std::vector<std::complex<double>>& sym_qlms);
