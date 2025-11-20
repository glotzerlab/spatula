list(APPEND spatula_sources
    "export-module.cc"
    "BondOrder.h"
    "PGOP.h" "PGOP.cc"
    "BOOSOP.h" "BOOSOP.cc"
    "data/Vec3.h" "data/Vec3.cc"
    "data/Quaternion.h" "data/Quaternion.cc"
    "optimize/Optimize.h" "optimize/Optimize.cc"
    "optimize/RandomSearch.h" "optimize/RandomSearch.cc"
    "optimize/StepGradientDescent.h" "optimize/StepGradientDescent.cc"
    "optimize/NoOptimization.h" "optimize/NoOptimization.cc"
    "optimize/Mesh.h" "optimize/Mesh.cc"
    "optimize/Union.h" "optimize/Union.cc"
    "optimize/export_optimize.h" "optimize/export_optimize.cc"
    "util/Metrics.h"
    "util/Util.h"
    "util/QlmEval.h" "util/QlmEval.cc"
)

pybind11_add_module(_spatula ${spatula_sources})

set_target_properties(_spatula PROPERTIES
    CXX_STANDARD 17
    CMAKE_CXX_STANDARD_REQUIRED True
    THREADS_PREFER_PTHREAD_FLAG ON
)
if (CMAKE_SYSTEM_NAME EQUAL "Linux")
    target_compile_options(_spatula PRIVATE -pthreads)
endif (CMAKE_SYSTEM_NAME EQUAL "Linux")
if (UNIX)
    target_link_libraries(_spatula PRIVATE Threads::Threads)
    target_compile_options(_spatula PRIVATE -Werror -Wall -Wextra)
endif (UNIX)
if (MSVC)
    # Does not compile without errors on MSVC
    target_compile_options(_spatula PRIVATE /Wall  /permissive-)
endif (MSVC)
target_compile_definitions(_spatula PRIVATE VERSION_INFO=${VERSION_INFO})

# Installs C++ extension into the root of the Python package
install(TARGETS _spatula LIBRARY DESTINATION spatula)
