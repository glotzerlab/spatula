nanobind_add_module(_spatula_nb
    export.cc
    export-threads.cc
    ../PGOP.cc
)

target_include_directories(
    _spatula_nb
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../util
    ${CMAKE_CURRENT_SOURCE_DIR}/../data
    ${CMAKE_CURRENT_SOURCE_DIR}/../optimize
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_target_properties(_spatula_nb PROPERTIES
    CXX_STANDARD 20
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
