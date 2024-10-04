#include "NoOptimization.h"

namespace pgop { namespace optimize {

NoOptimization::NoOptimization(const data::Vec3& initial_point)
    : Optimizer()
{
    // Set the initial point
    m_point = initial_point;
    m_terminate = false; // Start off not terminated
}

void NoOptimization::internal_next_point()
{
    // Perform one step (which does nothing) and set the termination flag
    m_terminate = true;
}

bool NoOptimization::terminate() const
{
    // Terminate after one step
    return m_terminate;
}

std::unique_ptr<Optimizer> NoOptimization::clone() const
{
    return std::make_unique<NoOptimization>(*this);
}

void export_no_optimization(py::module& m)
{
    py::class_<NoOptimization, Optimizer, std::shared_ptr<NoOptimization>>(
        m,
        "NoOptimization")
        .def(py::init<const data::Vec3&>());
}
}} // namespace pgop::optimize
