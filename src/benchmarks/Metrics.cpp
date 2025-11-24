#include <benchmark/benchmark.h>
#include "../util/Metrics.h"
#include "../data/Vec3.h"

#include <vector>
#include <complex>
#include <random>

// Helper function to generate random complex data for covariance benchmark
static std::vector<std::complex<double>> generate_complex_vector(size_t size) {
    std::vector<std::complex<double>> vec(size);
    // Use a fixed seed for reproducibility of test data
    std::mt19937 gen(1234);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (size_t i = 0; i < size; ++i) {
        vec[i] = {dis(gen), dis(gen)};
    }
    return vec;
}

// Benchmark for spatula::util::covariance
static void BM_Covariance(benchmark::State& state) {
    const size_t vec_size = state.range(0);
    auto f = generate_complex_vector(vec_size);
    auto g = generate_complex_vector(vec_size);

    for (auto _ : state) {
        // Run the function to benchmark
        benchmark::DoNotOptimize(spatula::util::covariance(f, g));
    }
    state.SetItemsProcessed(state.iterations());
    // 2 vectors of complex numbers
    state.SetBytesProcessed(state.iterations() * (2 * vec_size * sizeof(std::complex<double>)));
}
// Register the function as a benchmark
BENCHMARK(BM_Covariance)->Arg(100)->Arg(1000)->Arg(10000);

// Benchmark for spatula::util::compute_Bhattacharyya_coefficient_gaussian
static void BM_Bhattacharyya_Gaussian(benchmark::State& state) {
    spatula::data::Vec3 pos1(1.0, 2.0, 3.0);
    spatula::data::Vec3 pos2(1.1, 2.2, 3.3);
    double sigma1 = 0.5;
    double sigma2 = 0.6;

    for (auto _ : state) {
        benchmark::DoNotOptimize(spatula::util::compute_Bhattacharyya_coefficient_gaussian(pos1, pos2, sigma1, sigma2));
    }
}
BENCHMARK(BM_Bhattacharyya_Gaussian);


// Benchmark for spatula::util::compute_Bhattacharyya_coefficient_fisher
static void BM_Bhattacharyya_Fisher(benchmark::State& state) {
    spatula::data::Vec3 pos1(1.0, 0.0, 0.0);
    spatula::data::Vec3 pos2(0.9, 0.1, 0.0);
    double kappa1 = 10.0;
    double kappa2 = 12.0;

    for (auto _ : state) {
        benchmark::DoNotOptimize(spatula::util::compute_Bhattacharyya_coefficient_fisher(pos1, pos2, kappa1, kappa2));
    }
}
BENCHMARK(BM_Bhattacharyya_Fisher);

BENCHMARK_MAIN();
