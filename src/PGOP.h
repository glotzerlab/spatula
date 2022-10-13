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
template<typename distribution_type> class PGOP {
    public:
    PGOP(unsigned int max_l,
         const py::array_t<std::complex<double>> D_ij,
         std::shared_ptr<optimize::Optimizer>& optimizer,
         typename distribution_type::param_type distribution_params);

    py::tuple compute(const py::array_t<double> distances,
                      const py::array_t<double> weights,
                      const py::array_t<int> num_neighbors,
                      const unsigned int m,
                      const py::array_t<std::complex<double>> ylms,
                      const py::array_t<double> quad_positions,
                      const py::array_t<double> quad_weights) const;

    private:
    std::tuple<std::vector<double>, std::vector<data::Quaternion>>
    compute_particle(const std::vector<data::Vec3>& positions,
                     const std::vector<double>& weights,
                     const util::QlmEval& qlm_eval,
                     util::QlmBuf& qlm_buf,
                     unsigned int particle_index) const;

    std::tuple<double, data::Quaternion>
    compute_symmetry(const std::vector<data::Vec3>& positions,
                     const std::vector<double>& weights,
                     std::vector<data::Vec3>& rotated_distances_buf,
                     const std::vector<std::complex<double>>& D_ij,
                     const util::QlmEval& qlm_eval,
                     util::QlmBuf& qlm_buf,
                     unsigned int particle_index) const;

    double compute_pgop(const std::vector<double>& hsphere_pos,
                        const std::vector<data::Vec3>& position,
                        const std::vector<double>& weights,
                        std::vector<data::Vec3>& rotated_positions,
                        const std::vector<std::complex<double>>& D_ij,
                        const util::QlmEval& qlm_eval,
                        util::QlmBuf& qlm_buf) const;

    distribution_type m_distribution;
    unsigned int m_max_l;
    unsigned int m_n_symmetries;
    std::vector<std::vector<std::complex<double>>> m_Dij;
    std::shared_ptr<const optimize::Optimizer> m_optimize;
};

template<typename distribution_type> void export_pgop_class(py::module& m, const std::string& name);

void export_pgop(py::module& m);
} // End namespace pgop
