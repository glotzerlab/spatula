#include <benchmark/benchmark.h>
#include <cmath>
#include <random>
#include <vector>

#include "../data/Quaternion.h"
#include "../data/Vec3.h"
#include "../optimize/Mesh.h"
#include "../optimize/RandomSearch.h"

// Objective function for the optimizers to work on.
// This is a placeholder, so we make it fast
static double objective_function(const spatula::data::Vec3& p)
{
    return 0.0;
}

// Benchmark for RandomSearch optimizer
static void BM_RandomSearchOptimizer(benchmark::State& state)
{
    long unsigned int seed = 12345;
    for (auto _ : state) {
        state.PauseTiming();
        spatula::optimize::RandomSearch optimizer(state.range(0), seed);
        state.ResumeTiming();

        while (!optimizer.terminate()) {
            spatula::data::Vec3 p = optimizer.next_point();
            optimizer.record_objective(objective_function(p));
        }
        // To prevent the loop from being optimized away
        benchmark::DoNotOptimize(optimizer.get_optimum());
    }
}
BENCHMARK(BM_RandomSearchOptimizer)
    ->RangeMultiplier(10)
    ->Range(10, 10000)
    ->Unit(benchmark::kMicrosecond);

// Benchmark for Mesh optimizer
static void BM_MeshOptimizer(benchmark::State& state)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distrib(-1.0, 1.0);

    const auto num_points = state.range(0);
    std::vector<double> points(num_points * 4);
    for (long i = 0; i < num_points; ++i) {
        spatula::data::Quaternion q(distrib(gen), distrib(gen), distrib(gen), distrib(gen));
        q.normalize();
        points[i * 4 + 0] = q.w;
        points[i * 4 + 1] = q.x;
        points[i * 4 + 2] = q.y;
        points[i * 4 + 3] = q.z;
    }

    for (auto _ : state) {
        spatula::optimize::Mesh optimizer(points.data(), num_points);

        while (!optimizer.terminate()) {
            spatula::data::Vec3 p = optimizer.next_point();
            optimizer.record_objective(objective_function(p));
        }
        benchmark::DoNotOptimize(optimizer.get_optimum());
    }
}
BENCHMARK(BM_MeshOptimizer)->RangeMultiplier(10)->Range(10, 10000)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
