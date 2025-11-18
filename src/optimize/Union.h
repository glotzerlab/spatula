// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include "StepGradientDescent.h" // For StepGradientDescent in createFinalOptimizer
#include <functional>
#include <memory>
#include <stdexcept> // For std::runtime_error

#include "Optimize.h"

namespace spatula { namespace optimize {
/**
 * @brief Combine two optimizers into one optimization.
 */
class Union : public Optimizer {
    public:
    Union(const std::shared_ptr<const Optimizer>& initial_opt,
          std::function<std::unique_ptr<Optimizer>(const Optimizer&)> instantiate_final)
        : Optimizer(), m_inital_opt(initial_opt->clone()), m_final_opt(nullptr),
          m_instantiate_final(instantiate_final), m_on_final_opt(false)
    {
    }

    Union(const Union& original)
        : Optimizer(), m_inital_opt(original.m_inital_opt->clone()), m_final_opt(nullptr),
          m_instantiate_final(original.m_instantiate_final), m_on_final_opt(original.m_on_final_opt)
    {
        if (m_on_final_opt) {
            m_final_opt = original.m_final_opt->clone();
        }
    }

    ~Union() override = default;

    void record_objective(double objective) override
    {
        if (!m_need_objective) {
            throw std::runtime_error("Must get new point before recording objective.");
        }
        m_need_objective = false;
        m_objective = objective;
        if (objective < m_best_point.second) {
            m_best_point.first = m_point;
            m_best_point.second = objective;
        }
        getCurrentOptimizer().record_objective(objective);
    }

    void internal_next_point() override
    {
        if (!m_on_final_opt && getCurrentOptimizer().terminate()) {
            createFinalOptimizer();
            m_on_final_opt = true;
        }
        m_point = getCurrentOptimizer().next_point();
    }

    bool terminate() const override
    {
        return m_on_final_opt && getCurrentOptimizer().terminate();
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
}} // namespace spatula::optimize
