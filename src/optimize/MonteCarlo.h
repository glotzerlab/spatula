#pragma once

#include <memory>
#include <random>
#include <vector>

#include <pybind11/pybind11.h>

#include "Optimize.h"

namespace py = pybind11;

namespace pgop { namespace optimize {
class TrialMoveGenerator {
    public:
    TrialMoveGenerator(long unsigned int seed, double diameter, double kT, unsigned int dim);

    std::vector<double> generate();
    void generate(std::vector<double>& buf);

    bool accept(double neg_energy_change);

    private:
    std::vector<double> getSample();
    void getSample(std::vector<double>& buf);

    unsigned int m_dim;
    double m_diameter;
    std::mt19937_64 m_rng_engine;
    std::uniform_real_distribution<> m_uniform_dist;
    double m_inv_kT;
    std::uniform_real_distribution<> m_acceptance_dist;
};

class MonteCarlo : public Optimizer {
    public:
    MonteCarlo(const std::vector<double>& min_bounds,
               const std::vector<double>& max_bounds,
               const std::pair<std::vector<double>, double>& initial_point,
               double kT,
               double max_move_size,
               long unsigned int seed,
               unsigned int max_iter);

    ~MonteCarlo() override = default;

    void record_objective(double) override;
    std::vector<double> next_point() override;
    bool terminate() const override;
    std::pair<std::vector<double>, double> get_optimum() const override;
    std::unique_ptr<Optimizer> clone() const override;

    private:
    std::pair<std::vector<double>, double> m_best_point;
    std::pair<std::vector<double>, double> m_current_point;
    std::vector<double> m_trial_point;

    TrialMoveGenerator m_move_generator;
    std::vector<double> m_move_buf;
    double m_kT;

    unsigned int m_max_iter;
    unsigned int m_cnt;
};

void export_monte_carlo(py::module& m);
}} // namespace pgop::optimize
