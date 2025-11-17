// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include "BOOSOP.h"
#include "BondOrder.h"
#include "PGOP.h"
#include "data/Quaternion.h"
#include "optimize/Optimize.h"
#include "pybind11/complex.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "util/QlmEval.h"
#include "util/Util.h"

namespace py = pybind11;

namespace spatula {

template<typename distribution_type>
void export_BOOSOP_class(py::module& m, const std::string& name)
{
    py::class_<BOOSOP<distribution_type>>(m, name.c_str())
        .def(py::init<const py::array_t<std::complex<double>>,
                      std::shared_ptr<optimize::Optimizer>&,
                      typename distribution_type::param_type>())
        .def("compute", &BOOSOP<distribution_type>::compute)
        .def("refine", &BOOSOP<distribution_type>::refine);
}

void export_BOOSOP(py::module& m)
{
    export_BOOSOP_class<UniformDistribution>(m, "BOOSOPUniform");
    export_BOOSOP_class<FisherDistribution>(m, "BOOSOPFisher");
}

void export_PGOP(py::module& m)
{
    py::class_<PGOP>(m, "PGOP")
        .def(py::init<const py::list&,
                      std::shared_ptr<optimize::Optimizer>&,
                      const unsigned int,
                      bool>())
        .def("compute", &PGOP::compute);
}

} // namespace spatula
