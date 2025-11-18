nanobind_add_module(_spatula_nb
    export.cc
    export-threads.cc
    export-quat.cc
    export-optimizer.cc
    export-stores.cc
    export-PGOP.cc
    ../PGOP.cc
)

target_include_directories(_spatula_nb PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../util)

set_target_properties(_spatula_nb PROPERTIES
    CXX_STANDARD 17
    CMAKE_CXX_STANDARD_REQUIRED True
)
if (UNIX)
    target_compile_options(_spatula_nb PRIVATE -Werror -Wall -Wextra)
endif (UNIX)
if (MSVC)
    target_compile_options(_spatula_nb_metrics PRIVATE /Wall  /permissive-)
endif (MSVC)
target_compile_definitions(_spatula_nb PRIVATE VERSION_INFO=${VERSION_INFO})

install(TARGETS _spatula_nb LIBRARY DESTINATION spatula)
