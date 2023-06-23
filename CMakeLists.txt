cmake_minimum_required(VERSION 3.8)
project(_pgop)

set(DEFAULT_BUILD_TYPE Release)

# Add pthread library for thread_pool header
if (UNIX)
    find_package(Threads REQUIRED)
endif (UNIX)

find_package(pybind11 CONFIG REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/dependencies)

add_subdirectory(src)
