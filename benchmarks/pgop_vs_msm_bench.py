"""Benchmark comparing PGOP vs MSM runtime.

Run: python benchmarks/pgop_vs_msm_bench.py --data-file path/to/file.gsd
"""

# ruff: noqa: D103, B023
import argparse
import timeit
from pathlib import Path
import warnings

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=Path, required=True)
    args = parser.parse_args()

    SAMPLES = 1
    REPEATS = 1
    L = 6
    SYMMETRIES = ["Oh"]
    THREADS = [8, 4, 2, 1]

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

    COLORS = {"msm": "#2980B9", "pgop": "#00C9A4"}
    MARKER = {"msm": "H", "pgop": "o"}
    INFO = {"msm": f"Q{L}", "pgop": f"{SYMMETRIES}"}

    for method in ["msm", "pgop"]:
        data = [(r[1], r[2].mean(), r[2].std()) for r in results if r[0] == method]
        threads = [d[0] for d in data]
        means = [d[1] for d in data]
        stds = [d[2] for d in data]
        ax.errorbar(
            threads,
            means,
            yerr=stds,
            label=f"{method.upper()} : {INFO[method]}",
            color=COLORS[method],
            marker=MARKER[method],
            capsize=5,
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("Threads")
    ax.set_ylabel("Runtime (ms)")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(f"Runtime vs Threads (N={N_PARTICLES})")

    plt.tight_layout()
    plt.savefig("benchmark_results_ispc.svg", dpi=150)
