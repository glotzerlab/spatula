#pragma once
#include <memory>
#include <tuple>
#include <vector>

#include <complex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "data/Quaternion.h"
#include "optimize/Optimize.h"
#include "util/Metrics.h"
#include "util/QlmEval.h"
#include "util/Util.h"

namespace py = pybind11;

namespace pgop {

struct LocalNeighborhood {
    LocalNeighborhood(std::vector<data::Vec3>&& positions_, std::vector<double>&& weights_);

    void rotate(const data::Vec3& q);

    const std::vector<data::Vec3> positions;
    const std::vector<double> weights;
    std::vector<data::Vec3> rotated_positions;
};

class Neighborhoods {
    public:
    Neighborhoods(size_t N,
                  const int* neighbor_counts,
                  const double* weights,
                  const double* distance);

    LocalNeighborhood getNeighborhood(size_t i) const;
    std::vector<data::Vec3> getNormalizedDistances(size_t i) const;
    std::vector<double> getWeights(size_t i) const;
    int getNeighborCount(size_t i) const;

    private:
    const size_t m_N;
    const int* m_neighbor_counts;
    const double* m_distances;
    const double* m_weights;
    std::vector<size_t> m_neighbor_offsets;
};

struct PGOPStore {
    PGOPStore(size_t N_particles, size_t N_symmetries);
    size_t N_syms;
    py::array_t<double> op;
    py::array_t<double> rotations;

    void addOp(size_t i, const std::tuple<std::vector<double>, std::vector<data::Quaternion>>& op_);
    void addNull(size_t i);
    py::tuple getArrays();

    private:
    py::detail::unchecked_mutable_reference<double, 2> u_op;
    py::detail::unchecked_mutable_reference<double, 3> u_rotations;
};

template<typename distribution_type> class PGOP {
    public:
    PGOP(const py::array_t<std::complex<double>> D_ij,
         std::shared_ptr<optimize::Optimizer>& optimizer,
         typename distribution_type::param_type distribution_params);

    py::tuple compute(const py::array_t<double> distances,
                      const py::array_t<double> weights,
                      const py::array_t<int> num_neighbors,
                      const unsigned int m,
                      const py::array_t<std::complex<double>> ylms,
                      const py::array_t<double> quad_positions,
                      const py::array_t<double> quad_weights) const;

    py::array_t<double> refine(const py::array_t<double> distances,
                               const py::array_t<double> rotations,
                               const py::array_t<double> weights,
                               const py::array_t<int> num_neighbors,
                               const unsigned int m,
                               const py::array_t<std::complex<double>> ylms,
                               const py::array_t<double> quad_positions,
                               const py::array_t<double> quad_weights) const;

    private:
    std::tuple<std::vector<double>, std::vector<data::Quaternion>>
    compute_particle(LocalNeighborhood& neighborhood,
                     const util::QlmEval& qlm_eval,
                     util::QlmBuf& qlm_buf) const;
    std::tuple<double, data::Quaternion>
    compute_symmetry(LocalNeighborhood& neighborhood,
                     const std::vector<std::complex<double>>& D_ij,
                     const util::QlmEval& qlm_eval,
                     util::QlmBuf& qlm_buf) const;

    double compute_pgop(LocalNeighborhood& neighborhood,
                        const std::vector<std::complex<double>>& D_ij,
                        const util::QlmEval& qlm_eval,
                        util::QlmBuf& qlm_buf) const;

    void execute_func(std::function<void(size_t, size_t)> func, size_t N) const;

    distribution_type m_distribution;
    unsigned int m_n_symmetries;
    std::vector<std::vector<std::complex<double>>> m_Dij;
    std::shared_ptr<const optimize::Optimizer> m_optimize;
};

template<typename distribution_type> void export_pgop_class(py::module& m, const std::string& name);

void export_pgop(py::module& m);
} // End namespace pgop
