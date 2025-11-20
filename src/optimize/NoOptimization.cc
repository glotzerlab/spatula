// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include "../data/Quaternion.h"
#include "NoOptimization.h"

namespace spatula { namespace optimize {

NoOptimization::NoOptimization(const data::Quaternion& initial_point) : Optimizer()
{
    // Set the initial point
    m_point = initial_point.to_axis_angle_3D();
    m_terminate = false; // Start off not terminated
}

void NoOptimization::internal_next_point()
{
    // Perform one step (which does nothing) and set the termination flag
    m_terminate = true;
}

bool NoOptimization::terminate() const
{
    // Terminate after one step
    return m_terminate;
}

std::unique_ptr<Optimizer> NoOptimization::clone() const
{
    return std::make_unique<NoOptimization>(*this);
}

}} // namespace spatula::optimize
