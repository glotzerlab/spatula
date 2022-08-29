pybind11_add_module(_pgop export-module.cc
    BondOrder.h BondOrder.cc
    Metrics.h Metrics.cc
    Util.h Util.cc
    Optimize.h Optimize.cc
    PGOP.h PGOP.cc
    QlmEval.h QlmEval.cc
    Threads.h
    Weijer.h Weijer.cc
)

set_target_properties(_pgop PROPERTIES
    CXX_STANDARD 17
    CMAKE_CXX_STANDARD_REQUIRED True
    THREADS_PREFER_PTHREAD_FLAG ON
)
target_link_libraries(_pgop PRIVATE Threads::Threads)
target_compile_definitions(_pgop PRIVATE VERSION_INFO=${VERSION_INFO})
