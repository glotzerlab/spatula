pybind11_add_module(_pgop export-module.cc
    BondOrder.h BondOrder.cc
    PGOP.h PGOP.cc
    data/Vec3.h data/Vec3.cc
    data/Quaternion.h data/Quaternion.cc
    optimize/Optimize.h optimize/Optimize.cc
    optimize/RandomSearch.h optimize/RandomSearch.cc
    optimize/LineSearch.h optimize/LineSearch.cc
    optimize/StepGradientDescent.h optimize/StepGradientDescent.cc
    optimize/Mesh.h optimize/Mesh.cc
    optimize/Union.h optimize/Union.cc
    optimize/export_optimize.h optimize/export_optimize.cc
    util/Metrics.h util/Metrics.cc
    util/Util.h util/Util.cc
    util/QlmEval.h util/QlmEval.cc
    util/Threads.h util/Threads.cc
)

set_target_properties(_pgop PROPERTIES
    CXX_STANDARD 17
    CMAKE_CXX_STANDARD_REQUIRED True
    THREADS_PREFER_PTHREAD_FLAG ON
)
if (CMAKE_SYSTEM_NAME EQUAL "Linux")
    target_compile_options(_pgop PRIVATE -pthreads)
endif (CMAKE_SYSTEM_NAME EQUAL "Linux")
if (UNIX)
    target_link_libraries(_pgop PRIVATE Threads::Threads)
    target_compile_options(_pgop PRIVATE -Werror -Wall -Wextra)
endif (UNIX)
if (MSVC)
    # Does not compile without errors on MSVC
    target_compile_options(_pgop PRIVATE /Wall  /permissive-)
endif (MSVC)
target_compile_definitions(_pgop PRIVATE VERSION_INFO=${VERSION_INFO})
