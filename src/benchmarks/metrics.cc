// Copyright (c) 2021-2025 The Regents of the University of Michigan
// Part of spatula, released under the BSD 3-Clause License.

#include "../util/Metrics.h"
#include "../data/Vec3.h"
#include <benchmark/benchmark.h>

#include <random>
#include <vector>

// Benchmark for spatula::util::compute_Bhattacharyya_coefficient_gaussian
static void BM_Bhattacharyya_Gaussian(benchmark::State& state)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distrib(-5.0, 5.0);
    std::uniform_real_distribution<double> sigma_distrib(0.1, 1.0);

    struct Sample {
        spatula::data::Vec3 pos1;
        spatula::data::Vec3 pos2;
        double sigma1;
        double sigma2;
    };

    std::vector<Sample> samples(1024);
    for (auto& s : samples) {
        s.pos1 = {distrib(gen), distrib(gen), distrib(gen)};
        s.pos2 = {distrib(gen), distrib(gen), distrib(gen)};
        s.sigma1 = sigma_distrib(gen);
        s.sigma2 = sigma_distrib(gen);
    }

    size_t i = 0;
    for (auto _ : state) {
        const auto& s = samples[i++ % samples.size()];
        benchmark::DoNotOptimize(
            spatula::util::compute_Bhattacharyya_coefficient_gaussian(s.pos1,
                                                                      s.pos2,
                                                                      s.sigma1,
                                                                      s.sigma2));
    }
}
BENCHMARK(BM_Bhattacharyya_Gaussian);

// Benchmark for spatula::util::compute_Bhattacharyya_coefficient_fisher
static void BM_Bhattacharyya_Fisher(benchmark::State& state)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distrib(-1.0, 1.0);
    std::uniform_real_distribution<double> kappa_distrib(1.0, 20.0);

    struct Sample {
        spatula::data::Vec3 pos1;
        spatula::data::Vec3 pos2;
        double kappa1;
        double kappa2;
    };

    std::vector<Sample> samples(1024);
    for (auto& s : samples) {
        s.pos1 = {distrib(gen), distrib(gen), distrib(gen)};
        s.pos1.normalize();
        s.pos2 = {distrib(gen), distrib(gen), distrib(gen)};
        s.pos2.normalize();
        s.kappa1 = kappa_distrib(gen);
        s.kappa2 = kappa_distrib(gen);
    }

    size_t i = 0;
    for (auto _ : state) {
        const auto& s = samples[i++ % samples.size()];
        benchmark::DoNotOptimize(spatula::util::compute_Bhattacharyya_coefficient_fisher(s.pos1,
                                                                                         s.pos2,
                                                                                         s.kappa1,
                                                                                         s.kappa2));
    }
}
BENCHMARK(BM_Bhattacharyya_Fisher);

BENCHMARK_MAIN();
