cmake_minimum_required(VERSION 3.15...3.30)
project(spatula)

set(DEFAULT_BUILD_TYPE Release)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE ${DEFAULT_BUILD_TYPE})
endif()

# Add pthread library for thread_pool header
if (UNIX)
    find_package(Threads REQUIRED)
endif (UNIX)


find_package(pybind11 CONFIG REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/dependencies)
add_subdirectory(src)
add_subdirectory(spatula)
add_subdirectory(src/nanobind)

option(ENABLE_PROFILING "Enable profiling support" OFF)

if(ENABLE_PROFILING)
    message(STATUS "ENABLE_PROFILING is ON. Adding -pg to compiler and linker flags.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif()
