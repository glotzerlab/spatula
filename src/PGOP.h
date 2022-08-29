#pragma once
#include <complex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Metrics.h"
#include "QlmEval.h"
#include "Util.h"

template<typename distribution_type> class PGOP {
    public:
    PGOP(unsigned int max_l,
         const py::array_t<std::complex<double>> D_ij,
         std::unique_ptr<WeightedPNormBase> p_norm,
         typename distribution_type::param_type distribution_params);

    py::array_t<double> compute(const py::array_t<double> distances,
                                const py::array_t<int> num_neighbors,
                                const unsigned int m,
                                const py::array_t<std::complex<double>> ylms,
                                const py::array_t<double> quad_positions,
                                const py::array_t<double> quad_weights) const;

    private:
    std::vector<std::vector<double>> getDefaultRotations() const;

    std::vector<std::vector<double>> getInitialSimplex(const std::vector<double>& center) const;

    distribution_type getDistribution() const;

    std::vector<double> compute_particle(const std::vector<Vec3>::const_iterator& position_begin,
                                         const std::vector<Vec3>::const_iterator& position_end,
                                         const QlmEval& qlm_eval) const;

    std::vector<double> compute_pgop(const std::vector<double>& rotation,
                                     const std::vector<Vec3>::const_iterator& position_begin,
                                     const std::vector<Vec3>::const_iterator& position_end,
                                     std::vector<Vec3>& rotated_positions,
                                     std::vector<std::vector<std::complex<double>>>& sym_qlm_buf,
                                     const QlmEval& qlm_eval) const;

    double score(const std::vector<double>& pgop) const;

    typename distribution_type::param_type m_distribution_params;
    unsigned int m_max_l;
    unsigned int m_n_symmetries;
    std::vector<std::vector<std::complex<double>>> m_Dij;
    std::unique_ptr<WeightedPNormBase> m_p_norm;
    std::vector<std::vector<std::complex<double>>> m_sym_qlms;
    std::vector<std::complex<double>> m_qlms;
};

template<typename distribution_type> void export_pgop_class(py::module& m, const std::string& name);

void export_pgop(py::module& m);
