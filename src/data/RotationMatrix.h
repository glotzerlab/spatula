// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include "Vec3.h"
#include <array>

namespace spatula { namespace data {
/**
 * @brief Class provides helper methods for dealing with rotation matrices.
 *
 * spatula uses rotation matrices for high-throughput rotation of points.
 *
 * @tparam T The floating point type (float or double)
 */
template<typename T> struct RotationMatrix : std::array<T, 9> {
/**
 * @brief Rotate a Vec3 by a matrix R, returning a new vector.
 */
#ifdef __GNUC__ // GCC, Clang
    __attribute__((always_inline)) __attribute__((visibility("default")))
#endif
#ifdef _MSC_VER // MSVC
    __declspec(dllexport)
#endif
    inline Vec3<T> rotate(Vec3<T> vec) const
    {
        return Vec3<T>((*this)[0] * vec[0] + (*this)[1] * vec[1] + (*this)[2] * vec[2],
                       (*this)[3] * vec[0] + (*this)[4] * vec[1] + (*this)[5] * vec[2],
                       (*this)[6] * vec[0] + (*this)[7] * vec[1] + (*this)[8] * vec[2]);
    }

    inline static RotationMatrix<T> from_vec3(const Vec3<T>& v)
    {
        const auto angle = v.norm();
        if (std::abs(angle) < static_cast<T>(1e-7)) {
            return RotationMatrix<T> {static_cast<T>(1),
                                      static_cast<T>(0),
                                      static_cast<T>(0),
                                      static_cast<T>(0),
                                      static_cast<T>(1),
                                      static_cast<T>(0),
                                      static_cast<T>(0),
                                      static_cast<T>(0),
                                      static_cast<T>(1)};
        }
        const auto axis = v / angle;
        const T c {static_cast<T>(std::cos(angle))};
        const T s {static_cast<T>(std::sin(angle))};
        const T C = static_cast<T>(1.0) - c;
        const auto sv = axis * s;
        return RotationMatrix<T> {
            C * axis.x * axis.x + c,
            C * axis.x * axis.y - sv.z,
            C * axis.x * axis.z + sv.y,
            C * axis.y * axis.x + sv.z,
            C * axis.y * axis.y + c,
            C * axis.y * axis.z - sv.x,
            C * axis.z * axis.x - sv.y,
            C * axis.z * axis.y + sv.x,
            C * axis.z * axis.z + c,
        };
    }
};

// Typedefs for common precision types
using RotationMatrixd = RotationMatrix<double>;
using RotationMatrixf = RotationMatrix<float>;

}} // namespace spatula::data
