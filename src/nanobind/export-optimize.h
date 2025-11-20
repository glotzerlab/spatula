// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/memory.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "../optimize/Mesh.h"
#include "../optimize/NoOptimization.h"
#include "../optimize/Optimize.h"
#include "../optimize/RandomSearch.h"
#include "../optimize/StepGradientDescent.h"
#include "../optimize/Union.h"

namespace nb = nanobind;

namespace spatula { namespace optimize {

inline void export_mesh(nb::module_& m)
{
    nb::class_<Mesh, Optimizer>(m, "Mesh").def(
        nb::init<const std::vector<data::Quaternion>&>());
}

inline void export_no_optimization(nb::module_& m)
{
    nb::class_<NoOptimization, Optimizer>(m, "NoOptimization")
        .def(nb::init<const data::Quaternion&>());
}


/**
 * @brief Trampoline class for exposing Optimizer in Python.
 *
 * This shouldn't actually be used to extend the class but we need this to pass Optimizers through
 * from Python.
 */
class NbOptimizer : public Optimizer {
    public:
    using Optimizer::Optimizer;

    ~NbOptimizer() override = default;

    /// Get the next point to compute the objective for.
    void internal_next_point() override
    {
        NB_OVERRIDE_PURE(void, Optimizer, internal_next_point);
    }
    /// Record the objective function's value for the last querried point.
    void record_objective(double objective) override
    {
        NB_OVERRIDE(void, Optimizer, record_objective, objective);
    }
    /// Returns whether or not convergence or termination conditions have been met.
    bool terminate() const override
    {
        NB_OVERRIDE_PURE(bool, Optimizer, terminate);
    }

    /// Create a clone of this optimizer
    virtual std::unique_ptr<Optimizer> clone() const override
    {
        return std::make_unique<NbOptimizer>(*this);
    }
};

inline void export_base_optimize(nb::module_& m)
{
    nb::class_<Optimizer, NbOptimizer>(m, "Optimizer")
        .def("record_objective", &Optimizer::record_objective)
        .def_prop_ro("terminate", &Optimizer::terminate)
        .def_prop_ro("count", &Optimizer::getCount);
}

inline void export_random_search(nb::module_& m)
{
    nb::class_<RandomSearch, Optimizer>(m, "RandomSearch")
        .def(nb::init<unsigned int, unsigned int>())
        .def_prop_rw("max_iter", &RandomSearch::getIterations, &RandomSearch::setIterations)
        .def_prop_rw("seed", &RandomSearch::getSeed, &RandomSearch::setSeed);
}

inline void export_step_gradient_descent(nb::module_& m)
{
    nb::class_<StepGradientDescent, Optimizer>(
        m,
        "StepGradientDescent")
        .def(nb::init<const data::Quaternion&, unsigned int, double, double, double>());
}

inline void export_union(nb::module_& m)
{
    nb::class_<Union, Optimizer>(m, "Union")
        .def_static("with_step_gradient_descent",
                    [](const std::shared_ptr<const Optimizer> initial_opt,
                       unsigned int max_iter,
                       double initial_jump,
                       double learning_rate,
                       double tol) -> auto {
                        return std::make_shared<Union>(
                            initial_opt,
                            [max_iter, initial_jump, learning_rate, tol](const Optimizer& opt) {
                                return std::make_unique<StepGradientDescent>(
                                    data::Quaternion(opt.get_optimum().first),
                                    max_iter,
                                    initial_jump,
                                    learning_rate,
                                    tol);
                            });
                    });
}

inline void export_optimize(nb::module_& m)
{
    export_base_optimize(m);
    export_step_gradient_descent(m);
    export_mesh(m);
    export_random_search(m);
    export_union(m);
    export_no_optimization(m);
}
