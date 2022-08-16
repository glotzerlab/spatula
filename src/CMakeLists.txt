pybind11_add_module(_pgop export-module.cc
    BondOrder.h BondOrder.cc
    Util.h Util.cc
    Weijer.h Weijer.cc
)

target_compile_definitions(_pgop PRIVATE VERSION_INFO=${VERSION_INFO})
