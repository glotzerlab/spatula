#pragma once

#include <utility>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;

class Optimizer {
    public:
    Optimizer(const std::vector<double>& min_bounds, const std::vector<double>& max_bounds);

    virtual std::vector<double> next_point() = 0;
    virtual void record_objective(double);
    virtual bool terminate() const = 0;
    virtual std::pair<std::vector<double>, double> get_optimum() const = 0;

    protected:
    void clip_point(std::vector<double>& point);

    const std::vector<double> m_min_bounds;
    const std::vector<double> m_max_bounds;

    std::vector<double> m_point;
    double m_objective;

    bool m_need_objective;
};

class BruteForce : public Optimizer {
    public:
    BruteForce(const std::vector<std::vector<double>>& points,
               const std::vector<double>& min_bounds,
               const std::vector<double>& max_bounds);

    virtual void record_objective(double);
    virtual std::vector<double> next_point();
    virtual bool terminate() const;
    virtual std::pair<std::vector<double>, double> get_optimum() const;

    private:
    std::vector<std::vector<double>> m_points;
    size_t m_cnt;

    std::vector<double> m_best_point;
    double m_best_objective;
};

struct NelderMeadParams {
    double alpha;
    double gamma;
    double rho;
    double sigma;

    NelderMeadParams(double alpha_, double gamma_, double rho_, double sigma_);
};

class RollingStd {
    public:
    RollingStd();
    RollingStd(const std::vector<double>& values);
    void update(double new_value, double old_value);
    double std() const;
    double mean() const;

    private:
    double m_mean;
    double m_var;
    double m_n;
};

double compute_distance(const std::vector<double>& a, const std::vector<double>& b);

class OrderedSimplex {
    public:
    OrderedSimplex(unsigned int dim);

    void add(const std::vector<double>& point, const double objective);

    const std::vector<double>& get_point(size_t index) const;

    size_t size() const;

    double get_objective(size_t index) const;

    double get_objective_std() const;
    double get_objective_mean() const;
    double get_min_dist() const;

    std::vector<double> compute_centroid() const;

    const std::pair<std::vector<double>, double>& operator[](size_t index) const;

    private:
    void complete_initialization();
    void update_min_distance(const std::vector<double>& new_point);

    unsigned int m_dim;
    std::vector<std::pair<std::vector<double>, double>> m_points;
    RollingStd m_rolling_std;
    double m_min_dist;
};

class NelderMead : public Optimizer {
    public:
    NelderMead(NelderMeadParams params,
               std::vector<std::vector<double>>& initial_simplex,
               std::vector<double>& min_bounds,
               std::vector<double>& max_bounds,
               unsigned int max_iter,
               double m_dist_tol,
               double m_std_tol);

    virtual std::vector<double> next_point();
    virtual bool terminate() const;
    virtual std::pair<std::vector<double>, double> get_optimum() const;

    private:
    enum Stage {
        NEW_SIMPLEX = 0,
        REFLECT = 1,
        EXPAND = 2,
        OUTSIDE_CONTRACT = 3,
        INSIDE_CONTRACT = 4,
    };

    std::vector<double> reflect();

    std::vector<double> expand();

    std::vector<double> outside_contract();

    std::vector<double> inside_contract();

    std::vector<double> shrink();

    Stage m_stage;
    const NelderMeadParams m_params;
    const unsigned int m_dim;
    OrderedSimplex m_current_simplex;

    unsigned int m_max_iter;
    unsigned int m_iter;
    double m_dist_tol;
    double m_std_tol;

    std::pair<std::vector<double>, double> m_last_reflect;
    size_t m_new_simplex_index;
    std::vector<std::vector<double>> m_new_simplex;
};

void export_optimize(py::module& m);
