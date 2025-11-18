// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include "export-stores.h"

#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include "../stores.h"

namespace nb = nanobind;

namespace spatula {

void export_stores(nb::module_& m)
{
    nb::class_<BOOSOPStore>(m, "BOOSOPStore")
        .def(nb::init<size_t, size_t>(), nb::arg("N_particles"), nb::arg("N_symmetries"))
        .def_prop_ro("op",
                     [](BOOSOPStore& s) {
                         return nb::ndarray<double>(s.op.data(),
                                                    {(size_t)s.m_n_particles, (size_t)s.N_syms},
                                                    nb::cast(s));
                     })
        .def_prop_ro("rotations",
                     [](BOOSOPStore& s) {
                         return nb::ndarray<double>(
                             s.rotations.data(),
                             {(size_t)s.m_n_particles, (size_t)s.N_syms, (size_t)4},
                             nb::cast(s));
                     })
        .def("addOp", &BOOSOPStore::addOp, nb::arg("i"), nb::arg("op_"))
        .def("addNull", &BOOSOPStore::addNull, nb::arg("i"));

    nb::class_<PGOPStore>(m, "PGOPStore")
        .def(nb::init<size_t, size_t>(), nb::arg("N_particles"), nb::arg("N_symmetries"))
        .def_prop_ro("op",
                     [](PGOPStore& s) {
                         return nb::ndarray<double>(s.op.data(),
                                                    {(size_t)s.m_n_particles, (size_t)s.N_syms},
                                                    nb::cast(s));
                     })
        .def_prop_ro("rotations",
                     [](PGOPStore& s) {
                         return nb::ndarray<double>(
                             s.rotations.data(),
                             {(size_t)s.m_n_particles, (size_t)s.N_syms, (size_t)4},
                             nb::cast(s));
                     })
        .def("addOp", &PGOPStore::addOp, nb::arg("i"), nb::arg("op_"))
        .def("addNull", &PGOPStore::addNull, nb::arg("i"));
}

} // namespace spatula
