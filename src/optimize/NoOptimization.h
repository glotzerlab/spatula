// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include "../data/Quaternion.h"
#include "Optimize.h"

namespace spatula { namespace optimize {

class NoOptimization : public Optimizer {
    public:
    NoOptimization(const data::Quaternion& initial_point) : Optimizer()
    {
        // Set the initial point
        m_point = initial_point.to_axis_angle_3D();
        m_terminate = false; // Start off not terminated
    }

    // Implements internal_next_point but does nothing
    void internal_next_point() override
    {
        // Perform one step (which does nothing) and set the termination flag
        m_terminate = true;
    }

    // The terminate method will immediately terminate the optimization after one step
    bool terminate() const override
    {
        // Terminate after one step
        return m_terminate;
    }

    // Clone function for Pybind11
    std::unique_ptr<Optimizer> clone() const override
    {
        return std::make_unique<NoOptimization>(*this);
    }

    private:
    // Flag to terminate the optimization
    bool m_terminate;
};
}} // namespace spatula::optimize
