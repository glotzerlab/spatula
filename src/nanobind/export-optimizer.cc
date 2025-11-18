// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include "export-optimizer.h"

#include <nanobind/stl/vector.h>

#include "../data/Quaternion.h"
#include "../data/Vec3.h"
#include "../optimize/Mesh.h"
#include "../optimize/NoOptimization.h"
#include "../optimize/Optimize.h"
#include "../optimize/RandomSearch.h"
#include "../optimize/StepGradientDescent.h"
#include "../optimize/Union.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace spatula { namespace optimize {
void export_optimizers(nb::module_& m)
{
    nb::class_<Optimizer>(m, "Optimizer")
        .def("record_objective", &Optimizer::record_objective)
        .def_prop_ro("terminate", &Optimizer::terminate)
        .def_prop_ro("count", &Optimizer::getCount);

    nb::class_<Mesh, Optimizer>(m, "Mesh").def(nb::init<const std::vector<data::Quaternion>&>());

    nb::class_<RandomSearch, Optimizer>(m, "RandomSearch")
        .def(nb::init<unsigned int, unsigned int>(), "max_iter"_a, "seed"_a)
        .def_prop_rw("max_iter", &RandomSearch::getIterations, &RandomSearch::setIterations)
        .def_prop_rw("seed", &RandomSearch::getSeed, &RandomSearch::setSeed);

    nb::class_<StepGradientDescent, Optimizer>(m, "StepGradientDescent")
        .def(nb::init<const data::Vec3&, unsigned int, double, double, double>(),
             "initial_point"_a,
             "max_iter"_a,
             "initial_jump"_a,
             "learning_rate"_a,
             "tol"_a);

    nb::class_<Union, Optimizer>(m, "Union")
        .def_static(
            "with_step_gradient_descent",
            [](const std::shared_ptr<const Optimizer> initial_opt,
               unsigned int max_iter,
               double initial_jump,
               double learning_rate,
               double tol) -> std::shared_ptr<Union> {
                return std::make_shared<Union>(
                    initial_opt,
                    [max_iter, initial_jump, learning_rate, tol](const Optimizer& opt) {
                        return std::make_unique<StepGradientDescent>(opt.get_optimum().first,
                                                                     max_iter,
                                                                     initial_jump,
                                                                     learning_rate,
                                                                     tol);
                    });
            },
            "optimizer"_a,
            "max_iter"_a,
            "initial_jump"_a,
            "learning_rate"_a,
            "tol"_a);

    nb::class_<NoOptimization, Optimizer>(m, "NoOptimization").def(nb::init<const data::Vec3&>());
}
}} // namespace spatula::optimize
