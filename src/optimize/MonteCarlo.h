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

    void setSeed(long unsigned int seed);

    double getDiameter() const;
    void setDiameter(double diameter);

    void setkT(double kT);

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
    void internal_next_point() override;
    bool terminate() const override;
    std::unique_ptr<Optimizer> clone() const override;
    void specialize(unsigned int particle_index) override;

    double getkT() const;
    void setkT(double kT);

    long unsigned int getSeed() const;
    void setSeed(long unsigned int seed);

    unsigned int getIter() const;
    void setIter(unsigned int iter);

    double getMaxMoveSize() const;
    void setMaxMoveSize(double max_move_size);

    unsigned int getCount() const;

    private:
    std::pair<std::vector<double>, double> m_best_point;
    std::pair<std::vector<double>, double> m_current_point;

    long unsigned int m_seed;
    TrialMoveGenerator m_move_generator;
    std::vector<double> m_move_buf;
    double m_kT;

    unsigned int m_max_iter;
};

void export_monte_carlo(py::module& m);
}} // namespace pgop::optimize
