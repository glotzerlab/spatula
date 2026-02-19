"""Benchmark comparing PGOP vs MSM runtime using pyperf.

Run: python benchmarks/pgop_vs_msm_bench.py -o results.json
CSV: python -m pyperf dump results.json --csv > results.csv
"""

# ruff: noqa: D103
import warnings
from pathlib import Path

import freud
import numpy as np
import pyperf

import spatula

if (p := Path("results.json")).exists():
    p.unlink()
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
    runner = pyperf.Runner(processes=1)

    L = 6
    SYMMETRIES = ["Oh", "D3h"]

    for n_threads in [1, 2, 4, 8]:
        box, points = make_system(n_threads)
        voronoi = make_voronoi(box, points)

        freud.set_num_threads(n_threads)
        spatula.util.set_num_threads(n_threads)

        runner.timeit(
            name=f"msm: threads={n_threads}",
            stmt=f"compute_msm({L}, (box, points), voronoi)",
            setup="from __main__ import box, points, voronoi, compute_msm",
        )

        runner.timeit(
            name=f"pgop: threads={n_threads}",
            stmt=f"compute_pgop({SYMMETRIES}, (box, points), voronoi)",
            setup="from __main__ import box, points, voronoi, compute_pgop",
        )
