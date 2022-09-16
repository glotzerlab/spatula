#include <cmath>

#include "Util.h"

namespace pgop { namespace util {
// Assumes points are on the unit sphere
double fast_angle_eucledian(const Vec3& ref_x, const Vec3& x)
{
    return std::acos(ref_x.dot(x));
}

std::vector<double> compute_rotation_matrix(double alpha, double beta, double gamma)
{
    auto s1 {std::sin(alpha)}, s2 {std::sin(beta)}, s3 {std::sin(gamma)};
    auto c1 {std::cos(alpha)}, c2 {std::cos(beta)}, c3 {std::cos(gamma)};
    auto R = std::vector<double> {
        c2 * c3,
        -c2 * s3,
        s2,
        s1 * s2 * c3 + s3 * c1,
        c1 * c3 - s1 * s2 * s3,
        -s1 * c2,
        s1 * s3 - s2 * c1 * c3,
        s2 * s3 * c1 + c3 * s1,
        c1 * c2,
    };
    return R;
}

std::vector<double> compute_rotation_matrix(const std::vector<double> rotation)
{
    return compute_rotation_matrix(rotation[0], rotation[1], rotation[2]);
}

void single_rotate(const Vec3& x, Vec3& x_prime, const std::vector<double>& R)
{
    x_prime.x = R[0] * x.x + R[1] * x.y + R[2] * x.z;
    x_prime.y = R[3] * x.x + R[4] * x.y + R[5] * x.z;
    x_prime.z = R[6] * x.x + R[7] * x.y + R[8] * x.z;
};

void rotate_euler(const std::vector<Vec3>::const_iterator points_begin,
                  const std::vector<Vec3>::const_iterator points_end,
                  std::vector<Vec3>::iterator rotated_points_it,
                  double alpha,
                  double beta,
                  double gamma)
{
    const auto R = compute_rotation_matrix(alpha, beta, gamma);
    return rotate_matrix(points_begin, points_end, rotated_points_it, R);
}

void rotate_euler(std::vector<Vec3>::const_iterator points_begin,
                  std::vector<Vec3>::const_iterator points_end,
                  std::vector<Vec3>::iterator rotated_points_it,
                  const std::vector<double>& rotation)
{
    return rotate_euler(points_begin,
                        points_end,
                        rotated_points_it,
                        rotation[0],
                        rotation[1],
                        rotation[2]);
}

void rotate_matrix(std::vector<Vec3>::const_iterator points_begin,
                   std::vector<Vec3>::const_iterator points_end,
                   std::vector<Vec3>::iterator rotated_points_it,
                   const std::vector<double>& R)
{
    for (auto it = points_begin; it != points_end; ++it, ++rotated_points_it) {
        single_rotate(*it, *rotated_points_it, R);
    }
}

std::vector<double> linspace(double start, double end, unsigned int n, bool include_end)
{
    double delta;
    if (include_end) {
        delta = (end - start) / static_cast<double>(n);
    } else {
        delta = (end - start) / static_cast<double>(n + 1);
    }
    auto v = std::vector<double>();
    v.reserve(n);
    for (unsigned int i {0}; i < n; ++i) {
        v.push_back(start + (static_cast<double>(i) * delta));
    }
    return v;
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

}} // namespace pgop::util
