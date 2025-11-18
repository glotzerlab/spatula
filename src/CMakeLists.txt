list(APPEND spatula_sources
    "export-module.cc"
    "PGOP.cc"
    "BOOSOP.cc"
    "optimize/Optimize.cc"
    "optimize/RandomSearch.cc"
    "optimize/StepGradientDescent.cc"
    "optimize/NoOptimization.cc"
    "optimize/Mesh.cc"
    "optimize/Union.cc"
    "util/QlmEval.cc"
    "util/export-utils.cc"
)

pybind11_add_module(_spatula ${spatula_sources})

set_target_properties(_spatula PROPERTIES
    CXX_STANDARD 17
    CMAKE_CXX_STANDARD_REQUIRED True
    THREADS_PREFER_PTHREAD_FLAG ON
)
target_include_directories(_spatula PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/optimize)
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
