#pragma once

#include <complex>

std::vector<std::vector<std::complex<double>>>
symmetrize_qlms(std::vector<std::complex<double>> qlms,
                std::vector<std::vector<std::complex<double>>> D_ij,
                unsigned int max_l);
