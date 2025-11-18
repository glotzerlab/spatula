#include "export-optimizer.h"

#include <nanobind/stl/vector.h>

#include "../data/Quaternion.h"
#include "../optimize/Mesh.h"
#include "../optimize/Optimize.h"

namespace nb = nanobind;

namespace spatula { namespace optimize {
void export_mesh(nb::module_& m) {
    nb::class_<Mesh, Optimizer>(m, "Mesh")
        .def(nb::init<const std::vector<data::Quaternion>&>());
}
}}
