// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include "Optimize.h"

#include <pybind11/pybind11.h>

namespace spatula { namespace optimize {

class NoOptimization : public Optimizer {
    public:
    NoOptimization(const data::Vec3& initial_point);

    // Implements internal_next_point but does nothing
    void internal_next_point() override;

    // The terminate method will immediately terminate the optimization after one step
    bool terminate() const override;

    // Clone function for Pybind11
    std::unique_ptr<Optimizer> clone() const override;

    private:
    // Flag to terminate the optimization
    bool m_terminate;
};

void export_no_optimization(py::module& m);
}} // namespace spatula::optimize
