#include <pybind11/pybind11.h>

#include "Optimize.h"

namespace py = pybind11;

namespace pgop { namespace optimize {
class GradientDescent : public Optimizer {
    public:
    GradientDescent(const std::vector<double>& min_bounds,
                    const std::vector<double>& max_bounds,
                    const std::vector<double>& initial_point,
                    double alpha,
                    double max_move_size,
                    double tol,
                    unsigned int n_rounds);

    ~GradientDescent() override = default;

    void internal_next_point() override;
    bool terminate() const override;
    std::unique_ptr<Optimizer> clone() const override;

    unsigned int getNRounds() const;

    unsigned int getNRoundsMax() const;
    void setNRoundsMax(unsigned int n_rounds);

    unsigned int getCurrentDim() const;

    double getAlpha() const;
    void setAlpha(double alpha);

    double getMaxMoveSize() const;
    void setMaxMoveSize(double max_move_size);

    double getTol() const;
    void setTol(double tol);

    private:
    double getInitialDelta() const;

    std::pair<std::vector<double>, double> m_best_point;
    std::pair<std::vector<double>, double> m_last_point;
    unsigned int m_current_dim;
    unsigned int m_current_opt_count;
    unsigned int m_n_rounds;
    unsigned int m_n_rounds_max;
    double m_alpha;
    double m_max_move_size;
    double m_tol;
    double m_delta;
};

void export_gradient_descent(py::module& m);
}} // namespace pgop::optimize
