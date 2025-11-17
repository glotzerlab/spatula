// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include <pybind11/stl.h>

#include "Mesh.h"
#include "NoOptimization.h"
#include "Optimize.h"
#include "RandomSearch.h"
#include "StepGradientDescent.h"
#include "Union.h"
#include "export_optimize.h"

namespace spatula { namespace optimize {
void export_optimize(py::module& m)
{
    export_base_optimize(m);
    export_step_gradient_descent(m);
    export_mesh(m);
    export_random_search(m);
    export_union(m);
    export_no_optimization(m);
}

void export_mesh(py::module& m)
{
    py::class_<Mesh, Optimizer, std::shared_ptr<Mesh>>(m, "Mesh").def(
        py::init<const std::vector<data::Quaternion>&>());
}

void export_no_optimization(py::module& m)
{
    py::class_<NoOptimization, Optimizer, std::shared_ptr<NoOptimization>>(m, "NoOptimization")
        .def(py::init<const data::Vec3&>());
}

void export_base_optimize(py::module& m)
{
    py::class_<Optimizer, PyOptimizer, std::shared_ptr<Optimizer>>(m, "Optimizer")
        .def("record_objective", &Optimizer::record_objective)
        .def_property_readonly("terminate", &Optimizer::terminate)
        .def_property_readonly("count", &Optimizer::getCount);
}

void export_random_search(py::module& m)
{
    py::class_<RandomSearch, Optimizer, std::shared_ptr<RandomSearch>>(m, "RandomSearch")
        .def(py::init<unsigned int, unsigned int>())
        .def_property("max_iter", &RandomSearch::getIterations, &RandomSearch::setIterations)
        .def_property("seed", &RandomSearch::getSeed, &RandomSearch::setSeed);
}

void export_step_gradient_descent(py::module& m)
{
    py::class_<StepGradientDescent, Optimizer, std::shared_ptr<StepGradientDescent>>(
        m,
        "StepGradientDescent")
        .def(py::init<const data::Vec3&, unsigned int, double, double, double>());
}

void export_union(py::module& m)
{
    py::class_<Union, Optimizer, std::shared_ptr<Union>>(m, "Union")
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
                                    opt.get_optimum().first,
                                    max_iter,
                                    initial_jump,
                                    learning_rate,
                                    tol);
                            });
                    });
}
}} // namespace spatula::optimize
