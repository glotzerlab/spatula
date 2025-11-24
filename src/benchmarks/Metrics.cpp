#include "../util/Metrics.h"
#include "../data/Vec3.h"
#include <benchmark/benchmark.h>

#include <complex>
#include <random>
#include <vector>

// Benchmark for spatula::util::compute_Bhattacharyya_coefficient_gaussian
static void BM_Bhattacharyya_Gaussian(benchmark::State& state)
{
    spatula::data::Vec3 pos1(1.0, 2.0, 3.0);
    spatula::data::Vec3 pos2(1.1, 2.2, 3.3);
    double sigma1 = 0.5;
    double sigma2 = 0.6;

    for (auto _ : state) {
        benchmark::DoNotOptimize(
            spatula::util::compute_Bhattacharyya_coefficient_gaussian(pos1, pos2, sigma1, sigma2));
    }
}
BENCHMARK(BM_Bhattacharyya_Gaussian);

// Benchmark for spatula::util::compute_Bhattacharyya_coefficient_fisher
static void BM_Bhattacharyya_Fisher(benchmark::State& state)
{
    spatula::data::Vec3 pos1(1.0, 0.0, 0.0);
    spatula::data::Vec3 pos2(0.9, 0.1, 0.0);
    double kappa1 = 10.0;
    double kappa2 = 12.0;

    for (auto _ : state) {
        benchmark::DoNotOptimize(
            spatula::util::compute_Bhattacharyya_coefficient_fisher(pos1, pos2, kappa1, kappa2));
    }
}
BENCHMARK(BM_Bhattacharyya_Fisher);

BENCHMARK_MAIN();
