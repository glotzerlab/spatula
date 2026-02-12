// Copyright (c) 2021-2026 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include "../optimize/Mesh.h"
#include "../optimize/NoOptimization.h"
#include "../optimize/Optimize.h"
#include "../optimize/RandomSearch.h"
#include "../optimize/StepGradientDescent.h"
#include "../optimize/Union.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace spatula { namespace optimize {
void export_optimize(nb::module_& m)
{
    nb::class_<Optimizer>(m, "Optimizer")
        .def("record_objective", &Optimizer::record_objective)
        .def_prop_ro("terminate", &Optimizer::terminate)
        .def_prop_ro("count", &Optimizer::getCount);

    nb::class_<Mesh, Optimizer>(m, "Mesh").def(
        "__init__",
        [](Mesh* self,
           nb::ndarray<double, nb::shape<-1, 4>, nb::c_contig, nb::device::cpu> points) {
            new (self) Mesh(points.data(), points.shape(0));
        });

    nb::class_<RandomSearch, Optimizer>(m, "RandomSearch")
        .def(nb::init<unsigned int, unsigned int>(), "max_iter"_a, "seed"_a)
        .def_prop_rw("max_iter", &RandomSearch::getIterations, &RandomSearch::setIterations)
        .def_prop_rw("seed", &RandomSearch::getSeed, &RandomSearch::setSeed);

    nb::class_<StepGradientDescent, Optimizer>(m, "StepGradientDescent")
        .def(
            "__init__",
            [](StepGradientDescent* self,
               nb::tuple initial_point,
               unsigned int max_iter,
               float initial_jump,
               float learning_rate,
               float tol) {
                if (initial_point.size() != 4) {
                    throw std::invalid_argument(
                        "initial_point must be a tuple of 4 floats (w, x, y, z)");
                }
                const data::Quaternion q(nb::cast<float>(initial_point[0]),
                                         nb::cast<float>(initial_point[1]),
                                         nb::cast<float>(initial_point[2]),
                                         nb::cast<float>(initial_point[3]));
                new (self) StepGradientDescent(q, max_iter, initial_jump, learning_rate, tol);
            },
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
               float initial_jump,
               float learning_rate,
               float tol) -> std::shared_ptr<Union> {
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
            },
            "optimizer"_a,
            "max_iter"_a,
            "initial_jump"_a,
            "learning_rate"_a,
            "tol"_a);

    nb::class_<NoOptimization, Optimizer>(m, "NoOptimization").def(nb::init());
}

}} // namespace spatula::optimize
