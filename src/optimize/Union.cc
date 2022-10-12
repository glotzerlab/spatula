#include <pybind11/stl.h>

#include "BruteForce.h"
#include "MonteCarlo.h"
#include "NelderMead.h"
#include "Union.h"

namespace pgop { namespace optimize {
void export_union_optimizer(py::module& m)
{
    py::class_<Union, Optimizer, std::shared_ptr<Union>>(m, "Union")
        .def_static("brute_force_nelder_mead",
                    [](const std::shared_ptr<const BruteForce> initial_opt,
                       NelderMeadParams params,
                       const std::vector<double>& min_bounds,
                       const std::vector<double>& max_bounds,
                       unsigned int max_iter,
                       double dist_tol,
                       double std_tol,
                       double delta) {
                        return std::make_shared<Union>(
                            initial_opt,
                            min_bounds,
                            max_bounds,
                            [&](const Optimizer& brute_force) -> auto{
                                auto simplex = std::vector<std::vector<double>>();
                                const auto& best_point = brute_force.get_optimum().first;
                                for (size_t i {0}; i < simplex.size() + 1; ++i) {
                                    simplex.emplace_back(best_point);
                                }
                                for (size_t i {0}; i < simplex.size(); ++i) {
                                    simplex[i + 1][i] += delta;
                                }
                                return std::make_unique<NelderMead>(params,
                                                                    simplex,
                                                                    min_bounds,
                                                                    max_bounds,
                                                                    max_iter,
                                                                    dist_tol,
                                                                    std_tol);
                            });
                    })
        .def_static(
            "brute_force_mc",
            [](const std::shared_ptr<const BruteForce> initial_opt,
               const std::vector<double>& min_bounds,
               const std::vector<double>& max_bounds,
               double kT,
               double max_move_size,
               long unsigned int seed,
               unsigned int max_iter) -> auto{
                return std::make_shared<Union>(initial_opt,
                                               min_bounds,
                                               max_bounds,
                                               [&](const Optimizer& opt) {
                                                   return std::make_unique<MonteCarlo>(
                                                       min_bounds,
                                                       max_bounds,
                                                       opt.get_optimum(),
                                                       kT,
                                                       max_move_size,
                                                       seed,
                                                       max_iter);
                                               });
            });
}
}} // namespace pgop::optimize
