#!/bin/bash
# This script should be run from the project root
set -e

BENCHMARK_DIR="includes"
mkdir -p $BENCHMARK_DIR
BENCHMARK_CLONE_DIR="$BENCHMARK_DIR/benchmark"

# 1. Install Google Benchmark.
# ---------------------------
# Check if Google Benchmark is already installed.
if [ ! -d "$BENCHMARK_CLONE_DIR" ]; then
    # Clone the Google Benchmark repository.
    echo "Downloading Google Benchmark..."
    git clone https://github.com/google/benchmark.git "$BENCHMARK_CLONE_DIR"
fi

# Build Google Benchmark.
cd "$BENCHMARK_CLONE_DIR"
cmake -E make_directory "build"
cmake -E chdir "build" cmake -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON ../
cmake --build "build" --config Release
cd - > /dev/null # Go back to src/benchmarks

# 2. Build the benchmark.
# -----------------------
echo "Building benchmark..."
g++ -std=c++17 -isystem "$BENCHMARK_CLONE_DIR/include" -I./src \
  -L"$BENCHMARK_CLONE_DIR/build/src" -lbenchmark -lpthread \
  src/benchmarks/metrics.cc -o metrics_benchmark
g++ -std=c++17 -isystem "$BENCHMARK_CLONE_DIR/include" -I./src \
  -L"$BENCHMARK_CLONE_DIR/build/src" -lbenchmark -lpthread \
  src/benchmarks/optimizers.cc -o optimizers_benchmark

# 3. Run the benchmark.
# ---------------------
echo "Running metrics benchmark..."
./metrics_benchmark
echo "Running optimizers benchmark..."
./optimizers_benchmark
