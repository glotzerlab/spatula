// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <limits>
#include <tuple>
#include <vector>

#include "data/Quaternion.h"

namespace spatula {

struct BOOSOPStore {
    BOOSOPStore(size_t N_particles, size_t N_symmetries)
        : N_syms(N_symmetries), m_n_particles(N_particles), op(N_particles * N_symmetries),
          rotations(N_particles * N_symmetries * 4)
    {
    }

    size_t N_syms;
    size_t m_n_particles;
    std::vector<double> op;
    std::vector<double> rotations;

    void addOp(size_t i, const std::tuple<std::vector<double>, std::vector<data::Quaternion>>& op_)
    {
        const auto& values = std::get<0>(op_);
        const auto& rots = std::get<1>(op_);
        for (size_t j {0}; j < N_syms; ++j) {
            op[i * N_syms + j] = values[j];
            size_t rot_idx = i * N_syms * 4 + j * 4;
            rotations[rot_idx + 0] = rots[j].w;
            rotations[rot_idx + 1] = rots[j].x;
            rotations[rot_idx + 2] = rots[j].y;
            rotations[rot_idx + 3] = rots[j].z;
        }
    }

    void addNull(size_t i)
    {
        for (size_t j {0}; j < N_syms; ++j) {
            op[i * N_syms + j] = 0;
            size_t rot_idx = i * N_syms * 4 + j * 4;
            rotations[rot_idx + 0] = 1;
            rotations[rot_idx + 1] = 0;
            rotations[rot_idx + 2] = 0;
            rotations[rot_idx + 3] = 0;
        }
    }
};

struct PGOPStore {
    PGOPStore(size_t N_particles, size_t N_symmetries)
        : N_syms(N_symmetries), m_n_particles(N_particles), op(N_particles * N_symmetries),
          rotations(N_particles * N_symmetries * 4)
    {
    }

    size_t N_syms;
    size_t m_n_particles;
    std::vector<double> op;
    std::vector<double> rotations;

    void addOp(size_t i, const std::tuple<std::vector<double>, std::vector<data::Quaternion>>& op_)
    {
        const std::vector<double>& values = std::get<0>(op_);
        const std::vector<data::Quaternion>& rots = std::get<1>(op_);
        for (size_t j {0}; j < N_syms; ++j) {
            op[i * N_syms + j] = values[j];
            size_t rot_idx = i * N_syms * 4 + j * 4;
            rotations[rot_idx + 0] = rots[j].w;
            rotations[rot_idx + 1] = rots[j].x;
            rotations[rot_idx + 2] = rots[j].y;
            rotations[rot_idx + 3] = rots[j].z;
        }
    }

    void addNull(size_t i)
    {
        for (size_t j {0}; j < N_syms; ++j) {
            op[i * N_syms + j] = std::numeric_limits<double>::quiet_NaN(); // Set NaN
            size_t rot_idx = i * N_syms * 4 + j * 4;
            rotations[rot_idx + 0] = 1;
            rotations[rot_idx + 1] = 0;
            rotations[rot_idx + 2] = 0;
            rotations[rot_idx + 3] = 0;
        }
    }
};

} // namespace spatula
