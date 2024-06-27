#pragma once

#include <functional>
#include <memory>

#include <nanobind/nanobind.h>

#include "Optimize.h"

namespace pgop { namespace optimize {
namespace nb = nanobind;
/**
 * @brief Combine two optimizers into one optimization.
 */
class Union : public Optimizer {
    public:
    Union(const std::shared_ptr<const Optimizer>& initial_opt,
          std::function<std::unique_ptr<Optimizer>(const Optimizer&)> instantiate_final);

    Union(const Union& original);

    ~Union() override = default;

    void record_objective(double objective) override;

    void internal_next_point() override;

    bool terminate() const override;

    std::unique_ptr<Optimizer> clone() const override;

    private:
    Optimizer& getCurrentOptimizer();

    const Optimizer& getCurrentOptimizer() const;

    void createFinalOptimizer();

    std::unique_ptr<Optimizer> m_inital_opt;
    std::unique_ptr<Optimizer> m_final_opt;
    std::function<std::unique_ptr<Optimizer>(const Optimizer&)> m_instantiate_final;
    bool m_on_final_opt;
};

void export_union(nb::module& m);
}} // namespace pgop::optimize
