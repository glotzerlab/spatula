cmake_minimum_required(VERSION 3.26...3.30)
project(spatula LANGUAGES CXX)

# Option to disable ISPC (must be set before enable_language)
option(ENABLE_ISPC "Use ISPC for SIMD-optimized computations" ON)

if(ENABLE_ISPC)
    include(CheckLanguage)
    check_language(ISPC)
    if(CMAKE_ISPC_COMPILER)
        enable_language(ISPC)
        message(STATUS "ISPC enabled")
    else()
        message(WARNING "ISPC compiler not found, falling back to scalar implementations")
        set(ENABLE_ISPC OFF CACHE BOOL "Use ISPC for SIMD-optimized computations" FORCE)
    endif()
endif()

set(DEFAULT_BUILD_TYPE Release)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE ${DEFAULT_BUILD_TYPE})
endif()

# Add pthread library for thread_pool header
if (UNIX)
    find_package(Threads REQUIRED)
endif (UNIX)


find_package(
    Python 3.12
    COMPONENTS Interpreter Development.Module Development.SABIModule
    REQUIRED
)
if((NOT Python_INTERPRETER_ID STREQUAL "Python") OR (NOT TARGET Python::SABIModule))
    message(
        SEND_ERROR
        "Cannot activate stable ABI build: Interpreter ${Python_INTERPRETER_ID}, SABI: ${Python_SABI_LIBRARIES}"
    )
endif()
find_package(nanobind CONFIG REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/dependencies)
add_subdirectory(src)
add_subdirectory(spatula)
add_subdirectory(src/nanobind)

option(ENABLE_PROFILING "Enable profiling support" OFF)

if(ENABLE_PROFILING)
    message(STATUS "ENABLE_PROFILING is ON. Adding -g to compiler and linker flags.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")
endif()
