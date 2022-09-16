pybind11_add_module(_pgop export-module.cc
    BondOrder.h BondOrder.cc
    PGOP.h PGOP.cc
    data/Vec3.h data/Vec3.cc
    data/Quaternion.h data/Quaternion.cc
    optimize/Optimize.h optimize/Optimize.cc
    util/Metrics.h util/Metrics.cc
    util/Util.h util/Util.cc
    util/QlmEval.h util/QlmEval.cc
    util/Threads.h
)

set_target_properties(_pgop PROPERTIES
    CXX_STANDARD 20
    CMAKE_CXX_STANDARD_REQUIRED True
    THREADS_PREFER_PTHREAD_FLAG ON
)
target_link_libraries(_pgop PRIVATE Threads::Threads)
target_compile_options(_pgop PRIVATE -Werror -Wall -Wextra)
target_compile_definitions(_pgop PRIVATE VERSION_INFO=${VERSION_INFO})
