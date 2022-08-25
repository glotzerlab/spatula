#include <pybind11/pybind11.h>

#include "Optimize.h"
#include "PGOP.h"

PYBIND11_MODULE(_pgop, m)
{
    export_optimize(m);
    export_pgop(m);
}
