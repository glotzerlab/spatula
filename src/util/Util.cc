#include <cmath>
#include <iterator>
#include <numeric>
#include <ranges>

#include "Util.h"

namespace pgop { namespace util {
// Assumes points are on the unit sphere
double fast_angle_eucledian(const Vec3& ref_x, const Vec3& x)
{
    return std::acos(ref_x.dot(x));
}

void single_rotate(const Vec3& x, Vec3& x_prime, const std::vector<double>& R)
{
    x_prime.x = R[0] * x.x + R[1] * x.y + R[2] * x.z;
    x_prime.y = R[3] * x.x + R[4] * x.y + R[5] * x.z;
    x_prime.z = R[6] * x.x + R[7] * x.y + R[8] * x.z;
};

void symmetrize_qlm(const std::vector<std::complex<double>>& qlms,
                    const std::vector<std::complex<double>>& D_ij,
                    std::vector<std::complex<double>>& sym_qlm_buf,
                    unsigned int max_l)
{
    sym_qlm_buf.clear();
    sym_qlm_buf.reserve(qlms.size());
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

py::array_t<std::complex<double>>
wignerDSemidirectProduct(const py::array_t<std::complex<double>> D_a,
                         const py::array_t<std::complex<double>> D_b)
{
    auto u_D_a = D_a.unchecked<1>();
    auto u_D_b = D_b.unchecked<1>();
    size_t max_l = 0;
    size_t cnt = 0;
    while (cnt < static_cast<size_t>(u_D_a.size())) {
        max_l += 1;
        cnt += (2 * max_l + 1) * (2 * max_l + 1);
    }
    std::vector<size_t> l_skips;
    l_skips.emplace_back(0);
    auto l_sizes = std::views::iota(static_cast<size_t>(0), max_l)
                   | std::views::transform([](const auto& l) { return (2 * l + 1) * (2 * l + 1); });
    std::partial_sum(l_sizes.begin(), l_sizes.end(), std::back_inserter(l_skips));

    py::array_t<std::complex<double>> D_ab(u_D_a.size());
    auto u_D_ab = D_ab.mutable_unchecked<1>();
    for (size_t l {0}; l < max_l; ++l) {
        const size_t start_li = l_skips[l];
        const size_t max_m = 2 * l + 1;
        for (size_t m_prime {0}; m_prime < max_m; ++m_prime) {
            const size_t start_lmprime_i = start_li + m_prime * max_m;
            for (size_t m {0}; m < max_m; ++m) {
                std::complex<double> sum {0, 0};
                for (size_t m_prime_2 {0}; m_prime_2 < max_m; ++m_prime_2) {
                    sum += u_D_a(start_lmprime_i + m_prime_2)
                           * u_D_b(start_li + m_prime_2 * max_m + m);
                }
                u_D_ab(start_lmprime_i + m) = colapse_to_zero(sum, 1e-7);
            }
        }
    }
    return D_ab;
}

void export_util(py::module& m)
{
    m.def("wignerD_semidirect_prod", &wignerDSemidirectProduct);
}
}} // namespace pgop::util
