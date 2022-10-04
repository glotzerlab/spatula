#include <pybind11/stl.h>

#include "BruteForce.h"
#include "NelderMead.h"
#include "Optimize.h"
#include "export_optimize.h"

namespace pgop { namespace optimize {
void export_optimize(py::module& m)
{
    py::class_<Optimizer, PyOptimizer, std::shared_ptr<Optimizer>>(m, "Optimizer")
        .def("next_point", &Optimizer::next_point)
        .def("record_objective", &Optimizer::record_objective)
        .def_property_readonly("terminate", &Optimizer::terminate)
        .def_property_readonly("optimum", &Optimizer::get_optimum);

    py::class_<BruteForce, Optimizer, std::shared_ptr<BruteForce>>(m, "BruteForce")
        .def(py::init<const std::vector<std::vector<double>>&,
                      const std::vector<double>&,
                      const std::vector<double>&>());

    py::class_<NelderMeadParams>(m, "NelderMeadParams")
        .def(py::init<double, double, double, double>());

    py::class_<RollingStd>(m, "RollingStd")
        .def(py::init<std::vector<double>>())
        .def("update", &RollingStd::update)
        .def("std", &RollingStd::std)
        .def("mean", &RollingStd::mean);

    py::class_<OrderedSimplex>(m, "OrderedSimplex")
        .def(py::init<unsigned int>())
        .def("add", &OrderedSimplex::add)
        .def("size", &OrderedSimplex::size)
        .def("get_point", &OrderedSimplex::get_point)
        .def("get_objective", &OrderedSimplex::get_objective)
        .def("get_objective_std", &OrderedSimplex::get_objective_std)
        .def("get_objective_mean", &OrderedSimplex::get_objective_mean)
        .def("get_min_dist", &OrderedSimplex::get_min_dist)
        .def("compute_centroid", &OrderedSimplex::compute_centroid)
        .def("__getitem__", &OrderedSimplex::operator[]);

    py::class_<NelderMead, Optimizer, std::shared_ptr<NelderMead>>(m, "NelderMead")
        .def(py::init<NelderMeadParams,
                      std::vector<std::vector<double>>&,
                      std::vector<double>&,
                      std::vector<double>&,
                      unsigned int,
                      double,
                      double>());
}
}} // namespace pgop::optimize
