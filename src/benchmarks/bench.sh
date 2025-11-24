#!/bin/bash
# This script should be run from its containing directory: src/benchmarks/
set -e

# 1. Install Google Benchmark.
# ---------------------------
# Check if Google Benchmark is already installed.
if [ ! -d "benchmark" ]; then
    # Clone the Google Benchmark repository.
    echo "Downloading Google Benchmark..."
    git clone https://github.com/google/benchmark.git
    # Build Google Benchmark.
    cd benchmark
    cmake -E make_directory "build"
    cmake -E chdir "build" cmake -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON ../
    cmake --build "build" --config Release
    cd ..
fi

# 2. Build the benchmark.
# -----------------------
echo "Building benchmark..."
g++ -std=c++17 -isystem benchmark/include \
  -Lbenchmark/build/src -lbenchmark -lpthread \
  Metrics.cpp -o metrics_benchmark

# 3. Run the benchmark.
# ---------------------
echo "Running benchmark..."
./metrics_benchmark
