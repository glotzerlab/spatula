#include <vector>

#include "Weijer.h"

void symmetrize_qlms(const std::vector<std::complex<double>>& qlms,
                     const std::vector<std::vector<std::complex<double>>>& D_ij,
                     std::vector<std::vector<std::complex<double>>>& sym_qlm_buf,
                     unsigned int max_l)
{
    size_t num_syms = D_ij.size();

    for (size_t sym_i {0}; sym_i < num_syms; ++sym_i) {
        auto& sym_i_qlms = sym_qlm_buf[sym_i];
        sym_i_qlms.clear();
        const auto& d_ij = D_ij[sym_i];
        size_t qlm_i {0};
        size_t dij_index {0};
        for (size_t l {0}; l < max_l + 1; ++l) {
            size_t max_m {2 * l + 1};
            for (size_t m_prime {0}; m_prime < max_m; ++m_prime) {
                std::complex<double> sym_qlm {0.0, 0.0};
                for (size_t m {0}; m < max_m; ++m) {
                    sym_qlm += qlms[qlm_i + m] * d_ij[dij_index];
                    ++dij_index;
                }
                sym_i_qlms.emplace_back(sym_qlm);
            }
            qlm_i += max_m;
        }
    }
}
