#pragma once

#include <complex>
#include <vector>

void symmetrize_qlms(const std::vector<std::complex<double>>& qlms,
                     const std::vector<std::vector<std::complex<double>>>& D_ij,
                     std::vector<std::vector<std::complex<double>>>& sym_qlm_buf,
                     unsigned int max_l);
