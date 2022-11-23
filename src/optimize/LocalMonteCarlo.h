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
    TrialMoveGenerator(long unsigned int seed, double max_theta, double kT);

    data::Quaternion generate();

    bool accept(double neg_energy_change);

    void setSeed(long unsigned int seed);

    double getMaxTheta() const;
    void setMaxTheta(double diameter);

    void setkT(double kT);

    private:
    data::Vec3 getSample();
    void getSample(data::Vec3& buf);

    double m_max_theta;
    std::mt19937_64 m_rng_engine;
    std::uniform_real_distribution<> m_uniform_dist;
    double m_inv_kT;
    std::uniform_real_distribution<> m_acceptance_dist;
};

class MonteCarlo : public Optimizer {
    public:
    MonteCarlo(const std::pair<data::Quaternion, double>& initial_point,
               double kT,
               double max_theta,
               long unsigned int seed,
               unsigned int iterations);

    ~MonteCarlo() override = default;

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

    double getMaxTheta() const;
    void setMaxTheta(double max_move_size);

    unsigned int getCount() const;

    private:
    std::pair<data::Quaternion, double> m_current_point;

    long unsigned int m_seed;
    TrialMoveGenerator m_move_generator;
    double m_kT;

    unsigned int m_max_iter;
};

void export_monte_carlo(py::module& m);
}} // namespace pgop::optimize
