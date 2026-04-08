# Copyright (c) 2021-2026 The Regents of the University of Michigan
# Part of spatula, released under the BSD 3-Clause License.

r"""Generate benchmark data comparing PGOP vs MSM runtime.

Run:

```bash
python benchmarks/pgop_vs_msm_generate_data.py \
    --data-file path/to/file.gsd --output benchmarks/data/machine_name.txt
```
"""

# ruff: noqa: D103, B023, N806
import argparse
import timeit
import warnings
from pathlib import Path

import freud
import gsd.hoomd
import numpy as np

import spatula

warnings.filterwarnings("ignore")

# PGOP with 600 mesh points + gradient descent (full optimization)
OPTIMIZER_600 = spatula.optimize.Union.with_step_gradient_descent(
    spatula.optimize.Mesh.from_lattice(n_rotations=600)
)
# PGOP with 60 mesh points + gradient descent
OPTIMIZER_60 = spatula.optimize.Union.with_step_gradient_descent(
    spatula.optimize.Mesh.from_lattice(n_rotations=60)
)
# PGOP with no optimization (identity rotation only)
OPTIMIZER_NONE = spatula.optimize.NoOptimization()


def make_voronoi(box, points):
    voronoi = freud.locality.Voronoi()
    voronoi.compute((box, points))
    return voronoi


def compute_msm(ls: int | tuple[int], system, voronoi: freud.locality.Voronoi):
    msm = freud.order.Steinhardt(l=ls, weighted=True)
    return msm.compute(system, neighbors=voronoi.nlist)


def compute_pgop(symmetries, system, voronoi: freud.locality.Voronoi, optimizer):
    pgop = spatula.PGOP(symmetries, optimizer=optimizer, mode="full")
    return pgop.compute(system, sigmas=0.0175, neighbors=voronoi.nlist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=Path, required=True, help="Input GSD file")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file for benchmark data",
    )
    parser.add_argument(
        "--threads",
        type=str,
        default="96,64,32,16,8,4,2,1",
        help=(
            "Comma-separated list of thread counts to test. "
            "(default: 96,64,32,16,8,4,2,1)"
        ),
    )
    args = parser.parse_args()

    SAMPLES = 2
    REPEATS = 10
    L = 6
    SYMMETRIES = ["Oh"]
    THREADS = [int(t) for t in args.threads.split(",")]

    with gsd.hoomd.open(args.data_file) as f:
        frame = f[-1]
        box, points = (
            freud.Box(*frame.configuration.box),
            frame.particles.position,
        )
    N_PARTICLES = len(points)
    voronoi = make_voronoi(box, points)

    # Collect all results
    results = []

    for n_threads in THREADS:
        freud.set_num_threads(n_threads)
        spatula.util.set_num_threads(n_threads)

        msm_times = timeit.repeat(
            lambda: compute_msm(L, (box, points), voronoi),
            number=SAMPLES,
            repeat=REPEATS * 20,
        )
        msm_arr = np.array(msm_times) * 1000 / SAMPLES
        results.append(("msm", n_threads, msm_arr))

        pgop_times = timeit.repeat(
            lambda: compute_pgop(SYMMETRIES, (box, points), voronoi, OPTIMIZER_600),
            number=SAMPLES,
            repeat=REPEATS,
        )
        pgop_arr = np.array(pgop_times) * 1000 / SAMPLES
        results.append(("pgop", n_threads, pgop_arr))

        pgop_60_times = timeit.repeat(
            lambda: compute_pgop(SYMMETRIES, (box, points), voronoi, OPTIMIZER_60),
            number=SAMPLES,
            repeat=REPEATS,
        )
        pgop_60_arr = np.array(pgop_60_times) * 1000 / SAMPLES
        results.append(("pgop_60", n_threads, pgop_60_arr))

        pgop_none_times = timeit.repeat(
            lambda: compute_pgop(SYMMETRIES, (box, points), voronoi, OPTIMIZER_NONE),
            number=SAMPLES,
            repeat=REPEATS,
        )
        pgop_none_arr = np.array(pgop_none_times) * 1000 / SAMPLES
        results.append(("pgop_none", n_threads, pgop_none_arr))

    # Write results to output file in parseable format
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Path.open(args.output, "w") as f:
        f.write(f"N_PARTICLES={N_PARTICLES}\n")
        f.write(f"N_SAMPLES={SAMPLES}\n")
        for n_threads in THREADS:
            f.write(f"threads={n_threads}\n")
            for method in ["msm", "pgop", "pgop_60", "pgop_none"]:
                arr = next(
                    r[2] for r in results if r[0] == method and r[1] == n_threads
                )
                per_particle = arr / N_PARTICLES
                f.write(
                    f"  {method}: {arr.mean():.4f}ms +/- {arr.std():.4f}ms "
                    f"({per_particle.mean():.4f}ms/particle)\n"
                )

    print(f"Benchmark data written to {args.output}")
