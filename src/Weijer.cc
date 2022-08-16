#include <vector>

#include "Weijer.h"

namespace py = pybind11;

py::array_t<std::complex<double>> symmetrize_qlms(py::array_t<std::complex<double>> qlms,
                                                  py::array_t<std::complex<double>> Dij,
                                                  unsigned int max_l)
{
    const auto* u_qlms = static_cast<const std::complex<double>*>(qlms.data());
    const auto* u_Dij = static_cast<const std::complex<double>*>(Dij.data());

    size_t num_syms {static_cast<size_t>(Dij.shape(0))};
    const auto shape = std::vector<size_t> {num_syms, static_cast<size_t>(qlms.shape(0))};
    auto sym_qlms = py::array_t<std::complex<double>>(shape);
    auto* u_sym_qlms = static_cast<std::complex<double>*>(sym_qlms.mutable_data());

    size_t dij_i {0};
    size_t sym_qlm_i {0};
    for (size_t sym_i {0}; sym_i < num_syms; ++sym_i) {
        size_t qlm_i {0};
        for (size_t l {0}; l < max_l + 1; ++l) {
            size_t max_m {2 * l + 1};
            for (size_t m_prime {0}; m_prime < max_m; ++m_prime) {
                std::complex<double> sym_qlm {0.0, 0.0};
                for (size_t m {0}; m < max_m; ++m) {
                    sym_qlm += u_qlms[qlm_i + m] * u_Dij[dij_i];
                    ++dij_i;
                }
                u_sym_qlms[sym_qlm_i] = sym_qlm;
                ++sym_qlm_i;
            }
            qlm_i += max_m;
        }
    }
    return sym_qlms;
}

void export_weijer(py::module& m)
{
    m.def("symmetrize_qlms", &symmetrize_qlms);
}
