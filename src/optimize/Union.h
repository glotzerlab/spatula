#pragma once

#include <concepts>
#include <memory>

#include <pybind11/pybind11.h>

#include "Optimize.h"

namespace pgop { namespace optimize {
namespace py = pybind11;
/**
 * @brief Combine two optimizers into one optimization.
 */
class Union : public Optimizer {
    public:
    Union(const std::shared_ptr<const Optimizer>& initial_opt,
          const std::vector<double>& min_bounds,
          const std::vector<double>& max_bounds,
          std::function<std::unique_ptr<Optimizer>(const Optimizer&)> instantiate_final)
        : Optimizer(min_bounds, max_bounds), m_inital_opt(initial_opt->clone()),
          m_final_opt(nullptr), m_instantiate_final(instantiate_final), m_on_final_opt(false)
    {
    }

    Union(const Union& original)
        : Optimizer(original.m_min_bounds, original.m_max_bounds),
          m_inital_opt(original.m_inital_opt->clone()), m_final_opt(nullptr),
          m_instantiate_final(original.m_instantiate_final), m_on_final_opt(original.m_on_final_opt)
    {
        if (m_on_final_opt) {
            m_final_opt = original.m_final_opt->clone();
        }
    }

    ~Union() override = default;

    void record_objective(double objective) override
    {
        getCurrentOptimizer().record_objective(objective);
    }

    std::vector<double> next_point() override
    {
        if (!m_on_final_opt && getCurrentOptimizer().terminate()) {
            createFinalOptimizer();
            m_on_final_opt = true;
        }
        return getCurrentOptimizer().next_point();
    }

    bool terminate() const override
    {
        return !m_on_final_opt && getCurrentOptimizer().terminate();
    }

    std::pair<std::vector<double>, double> get_optimum() const override
    {
        return getCurrentOptimizer().get_optimum();
    }

    std::unique_ptr<Optimizer> clone() const override
    {
        return std::make_unique<Union>(*this);
    }

    private:
    Optimizer& getCurrentOptimizer()
    {
        if (m_on_final_opt) {
            return *m_final_opt.get();
        } else {
            return *m_inital_opt.get();
        }
    }

    const Optimizer& getCurrentOptimizer() const
    {
        if (m_on_final_opt) {
            return *m_final_opt.get();
        } else {
            return *m_inital_opt.get();
        }
    }
    void createFinalOptimizer()
    {
        m_final_opt = m_instantiate_final(*m_inital_opt.get());
    }

    std::unique_ptr<Optimizer> m_inital_opt;
    std::unique_ptr<Optimizer> m_final_opt;
    std::function<std::unique_ptr<Optimizer>(const Optimizer&)> m_instantiate_final;
    bool m_on_final_opt;
};

void export_union_optimizer(py::module& m);
}} // namespace pgop::optimize
