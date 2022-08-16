#include <pybind11/pybind11.h>

#include "Util.h"

PYBIND11_MODULE(_pgop, m)
{
    export_util(m);
}
