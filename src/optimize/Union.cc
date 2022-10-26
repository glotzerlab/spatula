#include <pybind11/stl.h>

#include "BruteForce.h"
#include "MonteCarlo.h"
#include "NelderMead.h"
#include "Union.h"

namespace pgop { namespace optimize {
Union::Union(const std::shared_ptr<const Optimizer>& initial_opt,
             const std::vector<double>& min_bounds,
             const std::vector<double>& max_bounds,
             std::function<std::unique_ptr<Optimizer>(const Optimizer&)> instantiate_final)
    : Optimizer(min_bounds, max_bounds), m_inital_opt(initial_opt->clone()), m_final_opt(nullptr),
      m_instantiate_final(instantiate_final), m_on_final_opt(false)
{
}

Union::Union(const Union& original)
    : Optimizer(original.m_min_bounds, original.m_max_bounds),
      m_inital_opt(original.m_inital_opt->clone()), m_final_opt(nullptr),
      m_instantiate_final(original.m_instantiate_final), m_on_final_opt(original.m_on_final_opt)
{
    if (m_on_final_opt) {
        m_final_opt = original.m_final_opt->clone();
    }
}

void Union::record_objective(double objective)
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

void Union::internal_next_point()
{
    if (!m_on_final_opt && getCurrentOptimizer().terminate()) {
        createFinalOptimizer();
        m_on_final_opt = true;
    }
    m_point = getCurrentOptimizer().next_point();
}

bool Union::terminate() const
{
    return m_on_final_opt && getCurrentOptimizer().terminate();
}

std::unique_ptr<Optimizer> Union::clone() const
{
    return std::make_unique<Union>(*this);
}

Optimizer& Union::getCurrentOptimizer()
{
    if (m_on_final_opt) {
        return *m_final_opt.get();
    } else {
        return *m_inital_opt.get();
    }
}

const Optimizer& Union::getCurrentOptimizer() const
{
    if (m_on_final_opt) {
        return *m_final_opt.get();
    } else {
        return *m_inital_opt.get();
    }
}

void Union::createFinalOptimizer()
{
    m_final_opt = m_instantiate_final(*m_inital_opt.get());
}

void export_union_optimizer(py::module& m)
{
    py::class_<Union, Optimizer, std::shared_ptr<Union>>(m, "Union")
        .def_static(
            "with_nelder_mead",
            [](const std::shared_ptr<const Optimizer> initial_opt,
               NelderMeadParams params,
               unsigned int max_iter,
               double dist_tol,
               double std_tol,
               double delta) {
                return std::make_shared<Union>(
                    initial_opt,
                    initial_opt->getMinBounds(),
                    initial_opt->getMaxBounds(),
                    [ params, max_iter, dist_tol, std_tol, delta ](const Optimizer& opt) -> auto{
                        auto simplex = std::vector<std::vector<double>>();
                        const auto& best_point = opt.get_optimum().first;
                        for (size_t i {0}; i < simplex.size() + 1; ++i) {
                            simplex.emplace_back(best_point);
                        }
                        for (size_t i {0}; i < simplex.size(); ++i) {
                            simplex[i + 1][i] += delta;
                        }
                        return std::make_unique<NelderMead>(params,
                                                            simplex,
                                                            opt.getMinBounds(),
                                                            opt.getMaxBounds(),
                                                            max_iter,
                                                            dist_tol,
                                                            std_tol);
                    });
            })
        .def_static(
            "with_mc",
            [](const std::shared_ptr<const Optimizer> initial_opt,
               double kT,
               double max_move_size,
               long unsigned int seed,
               unsigned int max_iter) -> auto{
                return std::make_shared<Union>(
                    initial_opt,
                    initial_opt->getMinBounds(),
                    initial_opt->getMaxBounds(),
                    [kT, max_move_size, seed, max_iter](const Optimizer& opt) {
                        return std::make_unique<MonteCarlo>(opt.getMinBounds(),
                                                            opt.getMaxBounds(),
                                                            opt.get_optimum(),
                                                            kT,
                                                            max_move_size,
                                                            seed,
                                                            max_iter);
                    });
            });
}
}} // namespace pgop::optimize
