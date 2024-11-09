#include <nanobind/nanobind.h>

#include "PGOP.h"
#include "data/Quaternion.h"
#include "data/Vec3.h"
#include "optimize/export_optimize.h"
#include "util/Threads.h"
#include "util/Util.h"

NB_MODULE(_pgop, m)
{
    pgop::data::export_Vec3(m);
    pgop::data::export_quaternion(m);
    pgop::optimize::export_optimize(m);
    pgop::export_pgop(m);
    pgop::util::export_threads(m);
    pgop::util::export_util(m);
}
