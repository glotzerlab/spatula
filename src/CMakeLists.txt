pybind11_add_module(_pgop export-module.cc  Util.h Util.cc)

target_compile_definitions(pgop PRIVATE VERSION_INFO=${VERSION_INFO})
