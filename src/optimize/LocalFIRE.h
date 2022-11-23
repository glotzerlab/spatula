#pragma once

#include <random>

#include <pybind11/pybind11.h>

#include "Optimize.h"

namespace pgop { namespace optimize {

class LocalFIRE : public Optimizer {
    public:
    LocalFIRE(const data::Quaternion& initial_point, unsigned int max_iter);
    ~LocalFIRE() override = default;
    /// Returns whether or not convergence or termination conditions have been met.
    bool terminate() const override;
    /// Create a clone of this optimizer
    std::unique_ptr<Optimizer> clone() const override;

    /// Set the next point to compute the objective for to m_point.
    void internal_next_point() override;

    private:
    enum Stage { GRADIENT = 1, SEARCH = 2 };
    void step();
    void findGradient();
    data::Quaternion computeNewRotation() const;
    void searchAlongGradient();

    unsigned int m_max_iter;
    unsigned int m_opt_cnt;
    Stage m_stage;
    unsigned short m_current_dim;
    std::pair<data::Quaternion, double> m_opt_point;
    data::Vec3 m_grad;
    std::tuple<data::Vec3, data::Vec3, data::Vec3> m_axes;
    double m_beta;
    double m_delta;
};

void export_localfire(py::module& m);
}} // namespace pgop::optimize
