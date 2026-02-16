# ISPC library for SIMD-optimized PGOP computations
add_library(spatula_ispc OBJECT
    ../computes/pgop_gaussian.ispc
)

# Detect ISPC architecture (see https://github.com/ispc/ispc/blob/main/examples/cpu/cmake/AddISPCExampleModern.cmake)
if(NOT ISPC_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "Intel")
        set(ispc_arch "x86-64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(ispc_arch "aarch64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
        set(ispc_arch "arm")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i686|i386")
        set(ispc_arch "x86")
    endif()
    set(ISPC_ARCH "${ispc_arch}" CACHE STRING "ISPC target architecture")
endif()

# ISPC will automatically select the correct SIMD instruction set if available
message(STATUS "ISPC architecture: ${ISPC_ARCH}, targets: ${ISPC_TARGETS}")
set_property(TARGET spatula_ispc PROPERTY ISPC_INSTRUCTION_SETS "${ISPC_TARGETS}")
target_compile_options(spatula_ispc PRIVATE "$<$<COMPILE_LANGUAGE:ISPC>:--arch=${ISPC_ARCH}>")

# Generate header file(s) for C++ integration
set_target_properties(spatula_ispc PROPERTIES
    ISPC_HEADER_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    POSITION_INDEPENDENT_CODE ON
)

nanobind_add_module(_spatula_nb
    export.cc
    export-threads.h
    export-boosop.cc
    ../BOOSOP.cc
    ../PGOP.cc
    ../util/QlmEval.cc
)

# Link ISPC object library
target_link_libraries(_spatula_nb PRIVATE spatula_ispc)

target_include_directories(
    _spatula_nb
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../util
    ${CMAKE_CURRENT_SOURCE_DIR}/../data
    ${CMAKE_CURRENT_SOURCE_DIR}/../optimize
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_BINARY_DIR}  # For generated ISPC header
)

set_target_properties(_spatula_nb PROPERTIES
    CXX_STANDARD 20
    CMAKE_CXX_STANDARD_REQUIRED True
)
if (UNIX)
    target_compile_options(_spatula_nb PRIVATE -Werror -Wall -Wextra)
endif (UNIX)
if (MSVC)
    target_compile_options(_spatula_nb PRIVATE /Wall  /permissive-)
endif (MSVC)
target_compile_definitions(_spatula_nb PRIVATE VERSION_INFO=${VERSION_INFO})

install(TARGETS _spatula_nb LIBRARY DESTINATION spatula)
