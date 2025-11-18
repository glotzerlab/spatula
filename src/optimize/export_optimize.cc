// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/make_iterator.h>

#include "Mesh.h"
#include "NoOptimization.h"
#include "Optimize.h"
#include "RandomSearch.h"
#include "StepGradientDescent.h"
#include "Union.h"
#include "export_optimize.h"

namespace nb = nanobind;

namespace spatula { namespace optimize {
void export_optimize(nb::module_& m)
{
    export_base_optimize(m);
    export_step_gradient_descent(m);
    export_mesh(m);
    export_random_search(m);
    export_union(m);
    export_no_optimization(m);
}

void export_mesh(nb::module_& m)
{
    nb::class_<Mesh, std::shared_ptr<Mesh>, Optimizer>(m, "Mesh").def(
        nb::init<const std::vector<data::Quaternion>&>());
}

void export_no_optimization(nb::module_& m)
{
    nb::class_<NoOptimization, std::shared_ptr<NoOptimization>, Optimizer>(m, "NoOptimization")
        .def(nb::init<const data::Vec3&>());
}

void export_base_optimize(nb::module_& m)
{
    nb::class_<Optimizer, std::shared_ptr<Optimizer>>(m, "Optimizer")
        .def("record_objective", &Optimizer::record_objective)
        .def_prop_ro("terminate", &Optimizer::terminate)
        .def_prop_ro("count", &Optimizer::getCount);
}

void export_random_search(nb::module_& m)
{
    nb::class_<RandomSearch, std::shared_ptr<RandomSearch>, Optimizer>(m, "RandomSearch")
        .def(nb::init<unsigned int, unsigned int>())
        .def_prop_rw("max_iter", &RandomSearch::getIterations, &RandomSearch::setIterations)
        .def_prop_rw("seed", &RandomSearch::getSeed, &RandomSearch::setSeed);
}

void export_step_gradient_descent(nb::module_& m)
{
    nb::class_<StepGradientDescent, std::shared_ptr<StepGradientDescent>, Optimizer>(
        m,
        "StepGradientDescent")
        .def(nb::init<const data::Vec3&, unsigned int, double, double, double>());
}

void export_union(nb::module_& m)
{
    nb::class_<Union, std::shared_ptr<Union>, Optimizer>(m, "Union")
        .def_static("with_step_gradient_descent",
                    [](const std::shared_ptr<const Optimizer> initial_opt,
                       unsigned int max_iter,
                       double initial_jump,
                       double learning_rate,
                       double tol) -> std::shared_ptr<Union> {
                        return std::make_shared<Union>(
                            initial_opt,
                            [max_iter, initial_jump, learning_rate, tol](const Optimizer& opt) {
                                return std::make_unique<StepGradientDescent>(
                                    opt.get_optimum().first,
                                    max_iter,
                                    initial_jump,
                                    learning_rate,
                                    tol);
                            });
                    });
}
}} // namespace spatula::optimize
