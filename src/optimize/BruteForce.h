#include <memory>
#include <vector>

#include "Optimize.h"

namespace pgop { namespace optimize {

/**
 * @brief An Optimizer that just tests prescribed points. The optimizer picks the best point out of
 * all provided test points.
 */
class BruteForce : public Optimizer {
    public:
    /**
     * @brief Create an BruteForce.
     *
     * All parameters are expected to have matching dimensions.
     *
     * @param points The points to test. Expected sizes are \f$ (N_{brute}, N_{dim} \f$.
     * @param min_bounds The minimum value allowed for each dimension. Set to
     * -std::numeric_limits<double>::infinity() for no minimum bounds. Expected dimension is
     * \f$ N_{dim} \f$.
     * @param max_bounds The maximum value allowed for each dimension. Set to
     * std::numeric_limits<double>::infinity() for no maximum bounds. Expected dimension is
     * \f$ N_{dim} \f$.
     */
    BruteForce(const std::vector<std::vector<double>>& points,
               const std::vector<double>& min_bounds,
               const std::vector<double>& max_bounds);

    ~BruteForce() override = default;

    void record_objective(double) override;
    std::vector<double> next_point() override;
    bool terminate() const override;
    std::pair<std::vector<double>, double> get_optimum() const override;
    std::unique_ptr<Optimizer> clone() const override;

    private:
    /// The set of points to evaluate.
    std::vector<std::vector<double>> m_points;
    /// The current iteration of the optimizer.
    size_t m_cnt;

    /// Current optimum point.
    std::vector<double> m_best_point;
    /// Current optimum objective.
    double m_best_objective;
};
}} // namespace pgop::optimize
