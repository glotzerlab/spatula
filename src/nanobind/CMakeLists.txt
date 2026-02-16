# ISPC library for SIMD-optimized PGOP computations
add_library(spatula_ispc OBJECT
    ../computes/pgop_gaussian.ispc
)

# Set ISPC target based on architecture using compile options
# The ISPC_TARGETS property doesn't work reliably, so we use target_compile_options
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "Intel")
    # Target AVX2 for x86_64 (8-wide float SIMD)
    target_compile_options(spatula_ispc PRIVATE "--target=avx2-i32x8")
    message(STATUS "x86_64 detected: ISPC targeting avx2-i32x8")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    # Use native ARM NEON target (4-wide float SIMD)
    target_compile_options(spatula_ispc PRIVATE "--target=neon-i32x4")
    message(STATUS "ARM detected: ISPC targeting neon-i32x4")
endif()

# Generate header file for C++ integration
set_target_properties(spatula_ispc PROPERTIES
    ISPC_HEADER_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
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
