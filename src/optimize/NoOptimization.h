// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include "Optimize.h"
#include "../data/Quaternion.h"
#include "../data/Vec3.h"

namespace spatula { namespace optimize {

class NoOptimization : public Optimizer
{
public:
    NoOptimization()
    {
        m_best_point.first = data::Quaternion(1.0, 0.0, 0.0, 0.0).to_axis_angle_3D();
    }

    NoOptimization(const data::Vec3& initial_point)
    {
        m_best_point.first = initial_point;
    }

    bool terminate() const override
    {
        return true;
    }

    std::unique_ptr<Optimizer> clone() const override
    {
        return std::make_unique<NoOptimization>(*this);
    }

    void internal_next_point() override { }
};

}} // namespace spatula::optimize
