"""Benchmark comparing PGOP vs MSM runtime.

Run: python benchmarks/pgop_vs_msm_bench.py
"""

# ruff: noqa: D103, B023
import timeit
import warnings

import freud
import numpy as np

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
    SAMPLES = 1
    REPEATS = 10
    L = 6
    SYMMETRIES = ["Oh"]  # , "D3h"]
    N_PARTICLES = 500
    THREADS = [8, 4, 2, 1]

    results = []
    box, points = make_system(N_PARTICLES)
    voronoi = make_voronoi(box, points)

    for n_threads in THREADS:
        print(f"threads={n_threads}")

        freud.set_num_threads(n_threads)
        spatula.util.set_num_threads(n_threads)

        msm_times = timeit.repeat(
            lambda: compute_msm(L, (box, points), voronoi),
            number=SAMPLES,
            repeat=REPEATS * 10,
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
    # print(results)
