#pragma once

#include <memory>
#include <vector>

#include "Optimize.h"

namespace pgop { namespace optimize {

/**
 * @brief An Optimizer that just tests prescribed points. The optimizer picks the best point out of
 * all provided test points.
 */
class Mesh : public Optimizer {
    public:
    /**
     * @brief Create an Mesh.
     *
     * All parameters are expected to have matching dimensions.
     *
     * @param points The points to test. Expected sizes are \f$ (N_{brute}, N_{dim} \f$.
     */
    Mesh(const std::vector<data::Quaternion>& points);

    ~Mesh() override = default;

    void internal_next_point() override;
    bool terminate() const override;
    std::unique_ptr<Optimizer> clone() const override;

    private:
    /// The set of points to evaluate.
    std::vector<data::Quaternion> m_points;
};

void export_mesh(py::module& m);
}} // namespace pgop::optimize
