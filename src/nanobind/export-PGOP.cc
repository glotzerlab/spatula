// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include "export-PGOP.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "../PGOP.h"
#include "../stores.h" // For PGOPStore

namespace nb = nanobind;
using namespace nb::literals;

namespace spatula {
// using Vector3f = nb::ndarray<float, nb::numpy, nb::shape<3>>;
using ArrayXXd = nb::ndarray<double, nb::numpy, nb::ndim<2>, nb::c_contig>;
using ArrayXX4d = nb::ndarray<double, nb::numpy, nb::shape<-1, -1, 4>, nb::c_contig>;

void export_pgop(nb::module_& m)
{
    nb::class_<PGOP>(m, "PGOP")
        .def(nb::init<const std::vector<std::vector<double>>&,
                      std::shared_ptr<optimize::Optimizer>&,
                      const unsigned int,
                      bool>(),
             nb::arg("R_ij"),
             nb::arg("optimizer"),
             nb::arg("mode"),
             nb::arg("compute_per_operator"))
        .def(
            "compute",
            [](const PGOP& pgop_instance,
               const nb::ndarray<double, nb::shape<-1, 3>, nb::c_contig> distances,
               const nb::ndarray<double, nb::ndim<1>, nb::c_contig> weights,
               const nb::ndarray<int, nb::ndim<1>, nb::c_contig> num_neighbors,
               const nb::ndarray<double, nb::ndim<1>, nb::c_contig> sigmas) {
                PGOPStore result = pgop_instance.compute(
                    num_neighbors.shape(0), // N_particles = num_query_points
                    distances.data(),
                    weights.data(),
                    num_neighbors.data(),
                    sigmas.data());

                size_t N_particles = result.m_n_particles;
                size_t N_symmetries = result.N_syms;

                // Create nanobind ndarray for 'op'
                // ArrayXXd op_array(
                //     result.op.data(),
                //     {N_particles, N_symmetries},
                //     deleter
                //     // nb::cast(std::move(result.op)) // Pass the moved vector as owner
                //     )
                //     .cast();

                auto op_array = ArrayXXd(result.op.data(), {N_particles, N_symmetries});
                auto rotations_array = ArrayXX4d(result.op.data(), {N_particles, N_symmetries, 4});

                // Create nanobind ndarray for 'rotations'
                // nb::ndarray<double, nb::ndim<3>, nb::c_contig> rotations_array(
                //     result.rotations.data(),
                //     {N_particles, N_symmetries, 4},
                //     nb::cast(std::move(result.rotations)) // Pass the moved vector as owner
                // );

                return nb::make_tuple(op_array, rotations_array);
            },
            nb::arg("distances"),
            nb::arg("weights"),
            nb::arg("num_neighbors"),
            nb::arg("sigmas"));
}

} // namespace spatula
