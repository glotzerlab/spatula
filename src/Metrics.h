#pragma once

#include <complex>
#include <vector>

class WeightedPNormBase {
    public:
    WeightedPNormBase();
    WeightedPNormBase(const std::vector<double>& weights);

    virtual double operator()(const std::vector<double>& vector) const = 0;

    protected:
    const std::vector<double> m_weights;
    const double m_normalization;
};

template<unsigned int p> class WeightedPNorm : public WeightedPNormBase {
    public:
    WeightedPNorm();
    WeightedPNorm(const std::vector<double>& weights);

    double operator()(const std::vector<double>& vector) const override;
};

std::vector<double> covariance(const std::vector<std::complex<double>>& qlms,
                               const std::vector<std::vector<std::complex<double>>>& sym_qlms);
