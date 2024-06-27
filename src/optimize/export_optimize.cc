#include <nanobind/nanobind.h>

#include "Mesh.h"
#include "Optimize.h"
#include "RandomSearch.h"
#include "StepGradientDescent.h"
#include "Union.h"
#include "export_optimize.h"

namespace pgop { namespace optimize {
void export_optimize(nb::module& m)
{
    export_base_optimize(m);
    export_step_gradient_descent(m);
    export_mesh(m);
    export_random_search(m);
    export_union(m);
}
}} // namespace pgop::optimize
