#include <pybind11/pybind11.h>

#include "Optimize.h"

namespace py = pybind11;

namespace pgop { namespace optimize {
class LocalBinaryOptimizer : public Optimizer {
    public:
    LocalBinaryOptimizer(const std::vector<double>& min_bounds,
                         const std::vector<double>& max_bounds,
                         const std::vector<double>& initial_point,
                         double max_move_size,
                         unsigned int iter_max);

    ~LocalBinaryOptimizer() override = default;

    void internal_next_point() override;
    bool terminate() const override;
    std::unique_ptr<Optimizer> clone() const override;

    unsigned int getIter() const;

    unsigned int getIterMax() const;
    void setIterMax(unsigned int iter_max);

    unsigned int getCurrentDim() const;

    private:
    enum class Stage { JACOBIAN = 1, MINIMIZE = 2 };

    void step();
    void findJacobian();
    double getInitialDelta() const;
    void normalizeGradient();
    void findContourMin();

    std::pair<std::vector<double>, double> m_last_point;
    Stage m_stage;
    bool m_terminate;
    unsigned int m_current_dim;
    std::vector<double> m_grad;
    double m_max_move_size;
    unsigned short m_opt_steps;
    double m_scale;
    std::pair<double, double> m_opt_objectives;
    unsigned short m_iter;
    unsigned short m_iter_max;
};

void export_local_binary(py::module& m);
}} // namespace pgop::optimize
