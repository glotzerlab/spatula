// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include "Vec3.h"
#include <array>

namespace spatula { namespace data {
/**
 * @brief Class provides helper methods for dealing with rotation matrices.
 *
 * spatula uses rotation matrices for high-throughput rotation of points.
 */
struct RotationMatrix : std::array<double, 9> {
    /**
     * @brief Rotate a Vec3 by a matrix R, returning a new vector.
     */
    __attribute__((always_inline)) __attribute__((visibility("default"))) inline Vec3
    rotate(Vec3 vec) const
    {
        return Vec3((*this)[0] * vec[0] + (*this)[1] * vec[1] + (*this)[2] * vec[2],
                    (*this)[3] * vec[0] + (*this)[4] * vec[1] + (*this)[5] * vec[2],
                    (*this)[6] * vec[0] + (*this)[7] * vec[1] + (*this)[8] * vec[2]);
    }
};
}} // namespace spatula::data
