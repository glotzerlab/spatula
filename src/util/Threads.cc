#include "Threads.h"

namespace pgop { namespace util {
void export_threads(nb::module_& m)
{
    m.def("set_num_threads",
          [](size_t num_threads) { ThreadPool::get().set_threads(num_threads); });
    m.def("get_num_threads", []() { return ThreadPool::get().get_num_threads(); });
}
}} // End namespace pgop::util
