"""Benchmark comparing PGOP vs MSM runtime.

Run: python benchmarks/pgop_vs_msm_bench.py --data-file path/to/file.gsd
"""

# ruff: noqa: D103, B023
import argparse
import timeit
import warnings
from pathlib import Path

import freud
import gsd.hoomd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import spatula

warnings.filterwarnings("ignore")

OPTIMIZER = spatula.optimize.Union.with_step_gradient_descent(
    spatula.optimize.Mesh.from_lattice(n_rotations=600)
)


def make_system(n):
    box = freud.box.Box.cube(10.0)
    rng = np.random.default_rng(42)
    points = rng.random((n, 3)) * 10.0
    return box, points


def make_voronoi(box, points):
    voronoi = freud.locality.Voronoi()
    voronoi.compute((box, points))
    return voronoi


def compute_msm(ls: int | tuple[int], system, voronoi: freud.locality.Voronoi):
    msm = freud.order.Steinhardt(l=ls, weighted=True)
    return msm.compute(system, neighbors=voronoi.nlist)


def compute_pgop(symmetries, system, voronoi: freud.locality.Voronoi):
    pgop = spatula.PGOP(symmetries, optimizer=OPTIMIZER, mode="full")
    return pgop.compute(system, sigmas=0.0175, neighbors=voronoi.nlist)


def parse_data_string(data: str) -> tuple[int, list[tuple[str, int, np.ndarray]]]:
    """Parse the DATA string into results format.

    Returns:
        N_PARTICLES and list of (method, n_threads, times_array) tuples
    """
    import re

    lines = data.strip().split("\n")
    n_particles = int(lines[0].split("=")[1])
    results = []

    current_threads = None
    for line in lines[1:]:
        if line.startswith("threads="):
            current_threads = int(line.split("=")[1])
        elif line.strip().startswith("msm:") or line.strip().startswith("pgop:"):
            match = re.match(
                r"\s*(msm|pgop):\s+([\d.]+)ms\s+\+/-\s+([\d.]+)ms\s+\(([\d.]+)ms/particle\)",
                line,
            )
            if match:
                method = match.group(1)
                mean_ms = float(match.group(2))
                std_ms = float(match.group(3))
                # Reconstruct a synthetic array with this mean and std
                # Use 20 samples (REPEATS default)
                times_arr = np.random.default_rng(42).normal(mean_ms, std_ms, 20)
                results.append((method, current_threads, times_arr))

    return n_particles, results


# Pre-computed benchmark data. Set to empty string to run benchmarks live.
DATA = """N_PARTICLES=13824
threads=128
  msm: 78.2841ms +/- 1.2453ms (0.0057ms/particle)
  pgop: 551.6669ms +/- 13.6764ms (0.0399ms/particle)
threads=64
  msm: 88.7745ms +/- 1.4662ms (0.0064ms/particle)
  pgop: 815.7703ms +/- 20.6883ms (0.0590ms/particle)
threads=32
  msm: 128.3489ms +/- 13.8206ms (0.0093ms/particle)
  pgop: 1404.6857ms +/- 30.7761ms (0.1016ms/particle)
threads=16
  msm: 153.0175ms +/- 7.7131ms (0.0111ms/particle)
  pgop: 2582.3495ms +/- 54.3781ms (0.1868ms/particle)
threads=8
  msm: 237.6860ms +/- 14.6902ms (0.0172ms/particle)
  pgop: 4853.8737ms +/- 8.1358ms (0.3511ms/particle)
threads=4
  msm: 136.0073ms +/- 8.8006ms (0.0098ms/particle)
  pgop: 9467.7100ms +/- 28.8863ms (0.6849ms/particle)
threads=2
  msm: 129.4901ms +/- 4.5500ms (0.0094ms/particle)
  pgop: 18664.1930ms +/- 53.5246ms (1.3501ms/particle)
threads=1
  msm: 184.5679ms +/- 0.1879ms (0.0134ms/particle)
  pgop: 36643.6964ms +/- 75.5736ms (2.6507ms/particle)
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=Path, required=False)
    args = parser.parse_args()

    SAMPLES = 1
    REPEATS = 20
    L = 6
    SYMMETRIES = ["Oh"]
    THREADS = [8, 4, 2, 1]

    if DATA:
        N_PARTICLES, results = parse_data_string(DATA)
    else:
        if args.data_file is None:
            parser.error("--data-file is required when DATA is not set")
        results = []
        with gsd.hoomd.open(args.data_file) as f:
            frame = f[-1]
            box, points = (
                freud.Box(*frame.configuration.box),
                frame.particles.position,
            )
        N_PARTICLES = len(points)
        print(f"N_PARTICLES={N_PARTICLES}")
        voronoi = make_voronoi(box, points)

        for n_threads in THREADS:
            print(f"threads={n_threads}")

            freud.set_num_threads(n_threads)
            spatula.util.set_num_threads(n_threads)

            msm_times = timeit.repeat(
                lambda: compute_msm(L, (box, points), voronoi),
                number=SAMPLES,
                repeat=REPEATS * 50,
            )
            msm_arr = np.array(msm_times) * 1000
            results.append(("msm", n_threads, msm_arr))
            msm_per_particle = msm_arr / N_PARTICLES
            print(
                f"  msm: {msm_arr.mean():.4f}ms +/- {msm_arr.std():.4f}ms "
                f"({msm_per_particle.mean():.4f}ms/particle)"
            )

            pgop_times = timeit.repeat(
                lambda: compute_pgop(SYMMETRIES, (box, points), voronoi),
                number=SAMPLES,
                repeat=REPEATS,
            )
            pgop_arr = np.array(pgop_times) * 1000  # ms
            results.append(("pgop", n_threads, pgop_arr))
            pgop_per_particle = pgop_arr / N_PARTICLES
            print(
                f"  pgop: {pgop_arr.mean():.4f}ms +/- {pgop_arr.std():.4f}ms "
                f"({pgop_per_particle.mean():.4f}ms/particle)"
            )

    # results is a list of (method, n_threads, times_array)
    # Access like: results[0][2] for the numpy array of times

    # Plotting
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots()

    # msm coloring is the freud pallette, #C7196A is from pgop paper figure 6
    COLORS = {"msm": "#2980B9", "pgop": "#C7196A"}
    MARKER = {"msm": "H", "pgop": "o"}
    INFO = {"msm": f"Q{L}", "pgop": f"{SYMMETRIES}"}

    # Get all thread counts for ideal scaling
    all_threads = sorted(set(r[1] for r in results))

    for method in ["msm", "pgop"]:
        data = [(r[1], r[2].mean(), r[2].std()) for r in results if r[0] == method]
        threads = [d[0] for d in data]
        # Convert ms/particle to particles/second: 1000 / (ms/particle)
        # Error propagation: std_y ≈ y² * std_x / 1000
        ms_per_particle = [d[1] / N_PARTICLES for d in data]
        std_ms_per_particle = [d[2] / N_PARTICLES for d in data]
        means = [1000 / x for x in ms_per_particle]
        stds = [(y**2) * sx / 1000 for y, sx in zip(means, std_ms_per_particle)]

        # Plot ideal scaling (1/nthreads) only for PGOP
        if method == "pgop":
            single_thread_data = [d for d in data if d[0] == 1]
            if single_thread_data:
                baseline_ms_per_particle = single_thread_data[0][1] / N_PARTICLES
                baseline_particles_per_sec = 1000 / baseline_ms_per_particle
                ideal_threads = [t for t in all_threads if t >= 1]
                ideal_means = [baseline_particles_per_sec * t for t in ideal_threads]
                ax.plot(
                    ideal_threads,
                    ideal_means,
                    linestyle="--",
                    color=COLORS[method],
                    alpha=0.5,
                    linewidth=1.5,
                )

        ax.errorbar(
            threads,
            means,
            yerr=stds,
            label=f"{method.upper()} : {INFO[method]}",
            color=COLORS[method],
            marker=MARKER[method],
            capsize=5,
            linewidth=2,
            markersize=5,
        )

    ax.set_xlabel("Threads")
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_threads)
    ax.set_ylabel("Particles per second")
    ax.set_yscale("linear")
    ax.legend()
    ax.set_title(f"Runtime vs Threads (N={N_PARTICLES})")

    plt.tight_layout()
    plt.savefig("benchmark_results_ispc.svg", dpi=150)
    print("plotted")

    # Print particles per second table
    print("\nParticles per second:")
    for method in ["msm", "pgop"]:
        print(f"  {method.upper()}:")
        data = [(r[1], r[2].mean(), r[2].std()) for r in results if r[0] == method]
        for n_threads, mean_ms, std_ms in sorted(data, key=lambda x: -x[0]):
            ms_per_particle = mean_ms / N_PARTICLES
            particles_per_sec = 1000 / ms_per_particle
            print(f"    threads={n_threads}: {particles_per_sec:.1f}")
