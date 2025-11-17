// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#pragma once

#include <random>

#include "Optimize.h"

namespace spatula { namespace optimize {

class StepGradientDescent : public Optimizer {
    /**
     * Optimize by rounds of 3 1D gradient descents.
     *
     * The representation is continuous in \f$ SO(3) \f$. StepGradientDescent
     * performs potentially multiple rounds of 3 1-dimensional gradient descents,
     * one for each dimension to find the local minimum. Each smaller optimization
     * is terminated when the improvement in objective is less than the provided
     * tolerance. The entire optimization ends when between rounds of optimization
     * the decrease in objective is less than the provided tolerance.
     *
     * The algorithm goes through a series of steps:
     * 1. Initialize - Only occurs in the first iteration
     * 2. Gradient - Find the gradient in a given dimension with a small jump
     * 3. Search - Perform the gradient descent in the current dimension
     */
    public:
    /**
     * @brief Create a StepGradientDescent optimizer.
     *
     * @param initial_point The starting point for the optimization.
     * @param max_iter Maximum number of iterations to run the algorithm.
     * @param initial_jump The jump size to use in determining the gradient for gradient descent.
     * @param learning_rate The learning rate for the gradient descent in the search stage.
     * @param tol The amount of improvement required to continue optimizing rather than terminating.
     */
    StepGradientDescent(const data::Vec3& initial_point,
                        unsigned int max_iter,
                        double initial_jump,
                        double learning_rate,
                        double tol);
    ~StepGradientDescent() override = default;
    /// Returns whether or not convergence or termination conditions have been met.
    bool terminate() const override;
    /// Create a clone of this optimizer
    std::unique_ptr<Optimizer> clone() const override;

    /// Set the next point to compute the objective for to m_point.
    void internal_next_point() override;

    private:
    enum Stage { INITIALIZE = 1, GRADIENT = 2, SEARCH = 4 };
    /// Switch the step of the algorithm
    void step();
    /// Run the first iteration and step afterwards
    void initialize();
    /// Find the gradient using a small jump and the forward difference
    void findGradient();
    /// Perform a gradient descent in the given direction
    void searchAlongGradient();

    // Hyperparameters
    /// Maximum number of iterations to run the algorithm
    unsigned int m_max_iter;
    /// The jump size to use in determining the gradient for gradient descent
    double m_initial_jump;
    /// The learning rate for the gradient descent in the search stage
    double m_learning_rate;
    /// The amount of improvement required to continue optimizing rather than terminating.
    double m_tol;
    // State variables
    // General
    /// The current stage of the optimization
    Stage m_stage;
    /// The starting objective for a given round of 1-dimensional optimizations
    double m_dim_starting_objective;
    /// Whether the optimization has terminated
    bool m_terminate;
    // Gradient Descent
    /// The current dimension for optimizing
    unsigned short m_current_dim;
    /// The last step's objective in gradient descent
    double m_last_objective;
    /// The current change between iterations in gradient descent.
    double m_delta;
};
}} // namespace spatula::optimize
