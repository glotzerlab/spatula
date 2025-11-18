#pragma once

#include <nanobind/nanobind.h>

namespace spatula { namespace optimize {
void export_optimizers(nanobind::module_& m);
}}
