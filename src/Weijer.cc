#include <vector>

#include "Weijer.h"

void symmetrize_qlms(const std::vector<std::complex<double>>& qlms,
                     const std::vector<std::vector<std::complex<double>>>& D_ijs,
                     std::vector<std::vector<std::complex<double>>>& sym_qlm_buf,
                     unsigned int max_l)
{
    for (size_t sym_i {0}; sym_i < D_ijs.size(); ++sym_i) {
        symmetrize_qlm(qlms, D_ijs[sym_i], sym_qlm_buf[sym_i], max_l);
    }
}

void symmetrize_qlm(const std::vector<std::complex<double>>& qlms,
                    const std::vector<std::complex<double>>& D_ij,
                    std::vector<std::complex<double>>& sym_qlm_buf,
                    unsigned int max_l)
{
    sym_qlm_buf.clear();
    size_t qlm_i {0};
    size_t dij_index {0};
    for (size_t l {0}; l < max_l + 1; ++l) {
        const size_t max_m {2 * l + 1};
        for (size_t m_prime {0}; m_prime < max_m; ++m_prime) {
            std::complex<double> sym_qlm {0.0, 0.0};
            for (size_t m {0}; m < max_m; ++m) {
                sym_qlm += qlms[qlm_i + m] * D_ij[dij_index];
                ++dij_index;
            }
            sym_qlm_buf.emplace_back(sym_qlm);
        }
        qlm_i += max_m;
    }
}
