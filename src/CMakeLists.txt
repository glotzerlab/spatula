pybind11_add_module(_pgop export-module.cc
    BondOrder.h BondOrder.cc
    Metrics.h Metrics.cc
    Util.h Util.cc
    Optimize.h Optimize.cc
    PGOP.h PGOP.cc
    QlmEval.h QlmEval.cc
    Weijer.h Weijer.cc
)

target_compile_definitions(_pgop PRIVATE VERSION_INFO=${VERSION_INFO})
