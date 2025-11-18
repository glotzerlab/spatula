// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include "export-stores.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "stores.h"

namespace py = pybind11;

namespace spatula {

void export_stores(py::module& m) {
    py::class_<BOOSOPStore>(m, "BOOSOPStore")
        .def(py::init<size_t, size_t>(),
             py::arg("N_particles"), py::arg("N_symmetries"))
        .def_property_readonly("op", [](BOOSOPStore& s) {
            return py::array_t<double>({(ssize_t)s.m_n_particles, (ssize_t)s.N_syms}, s.op.data(), py::cast(s));
        })
        .def_property_readonly("rotations", [](BOOSOPStore& s) {
            return py::array_t<double>({(ssize_t)s.m_n_particles, (ssize_t)s.N_syms, (ssize_t)4}, s.rotations.data(), py::cast(s));
        })
        .def("addOp", &BOOSOPStore::addOp,
             py::arg("i"), py::arg("op_"))
        .def("addNull", &BOOSOPStore::addNull,
             py::arg("i"));

    py::class_<PGOPStore>(m, "PGOPStore")
        .def(py::init<size_t, size_t>(),
             py::arg("N_particles"), py::arg("N_symmetries"))
        .def_property_readonly("op", [](PGOPStore& s) {
            return py::array_t<double>({(ssize_t)s.m_n_particles, (ssize_t)s.N_syms}, s.op.data(), py::cast(s));
        })
        .def_property_readonly("rotations", [](PGOPStore& s) {
            return py::array_t<double>({(ssize_t)s.m_n_particles, (ssize_t)s.N_syms, (ssize_t)4}, s.rotations.data(), py::cast(s));
        })
        .def("addOp", &PGOPStore::addOp,
             py::arg("i"), py::arg("op_"))
        .def("addNull", &PGOPStore::addNull,
             py::arg("i"));
}

} // namespace spatula
