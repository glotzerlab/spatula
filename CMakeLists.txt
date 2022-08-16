cmake_minimum_required(VERSION 3.0)
project(_pgop)

set(DEFAULT_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(pybind11 CONFIG REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/src)

add_subdirectory(src)
