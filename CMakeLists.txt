cmake_minimum_required(VERSION 3.15)
project(pgop)

set(DEFAULT_BUILD_TYPE Release)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE ${DEFAULT_BUILD_TYPE})
endif()

# Add pthread library for thread_pool header
if (UNIX)
    find_package(Threads REQUIRED)
endif (UNIX)


find_package(pybind11 CONFIG REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/dependencies)
add_subdirectory(src)
add_subdirectory(pgop)
