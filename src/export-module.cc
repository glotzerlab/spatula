#include <pybind11/pybind11.h>

#include "Optimize.h"
#include "PGOP.h"

PYBIND11_MODULE(_pgop, m)
{
    pgop::optimize::export_optimize(m);
    pgop::export_pgop(m);
}
