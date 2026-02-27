# Copyright (c) 2021-2026 The Regents of the University of Michigan
# Part of spatula, released under the BSD 3-Clause License.

r"""Generate benchmark data comparing PGOP runtime across different symmetry groups.

Run:
```bash
python benchmarks/group_size_scaling.py \
    --data-file path/to/file.gsd --output benchmarks/data/group_size_scaling.txt
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
from spatula.representations import CartesianRepMatrix

warnings.filterwarnings("ignore")

OPTIMIZER = spatula.optimize.NoOptimization()

# All point groups to benchmark, organized by family
# Orders are computed dynamically from CartesianRepMatrix
POINT_GROUPS = [
    # C family (low order)
    "Cs",
    "Ch",
    "Ci",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C2h",
    "C3h",
    "C4h",
    "C2v",
    "C3v",
    "C4v",
    "C5v",
    "C6v",
    # S family
    "S4",
    "S6",
    # D family
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D2h",
    "D3h",
    "D4h",
    "D2d",
    "D3d",
    # Polyhedral groups
    "T",
    "Td",
    "Th",
    "O",
    "Oh",
]


def get_symmetry_order(symmetry: str) -> int:
    """Get the order (number of operations) for a symmetry group."""
    return len(CartesianRepMatrix(symmetry).matrices)


def make_voronoi(box, points):
    voronoi = freud.locality.Voronoi()
    voronoi.compute((box, points))
    return voronoi


def compute_pgop(symmetry, system, voronoi: freud.locality.Voronoi):
    pgop = spatula.PGOP([symmetry], optimizer=OPTIMIZER, mode="full")
    return pgop.compute(system, sigmas=0.0175, neighbors=voronoi.nlist)


def compute_msm(l: int, system, voronoi: freud.locality.Voronoi):
    msm = freud.order.Steinhardt(l=l, weighted=True)
    return msm.compute(system, neighbors=voronoi.nlist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        type=Path,
        required=True,
        help="Input GSD file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file for benchmark data",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads (default: 1)",
    )
    args = parser.parse_args()

    SAMPLES = 1
    REPEATS = 10
    MSM_L_VALUES = [6, 8, 12]

    with gsd.hoomd.open(args.data_file) as f:
        frame = f[-1]
        box, points = (
            freud.Box(*frame.configuration.box),
            frame.particles.position,
        )
    N_PARTICLES = len(points)

    voronoi = make_voronoi(box, points)
    freud.set_num_threads(args.threads)
    spatula.util.set_num_threads(args.threads)

    # Collect results
    results = []

    # Benchmark MSM for each L value
    for l in MSM_L_VALUES:
        msm_times = timeit.repeat(
            lambda l=l: compute_msm(l, (box, points), voronoi),
            number=SAMPLES,
            repeat=REPEATS * 50,
        )
        msm_times_ms = np.array(msm_times) * 1000  # ms
        msm_particles_per_sec_arr = N_PARTICLES * 1000 / msm_times_ms
        results.append((f"msm_{l}", 0, msm_particles_per_sec_arr))  # order=0 for MSM

    for symmetry in POINT_GROUPS:
        order = get_symmetry_order(symmetry)
        times = timeit.repeat(
            lambda: compute_pgop(symmetry, (box, points), voronoi),
            number=SAMPLES,
            repeat=REPEATS,
        )
        times_ms = np.array(times) * 1000  # ms
        particles_per_sec_arr = N_PARTICLES * 1000 / times_ms
        results.append((symmetry, order, particles_per_sec_arr))

    # Write results to output file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Path.open(args.output, "w") as f:
        f.write(f"N_PARTICLES={N_PARTICLES}\n")
        f.write(f"THREADS={args.threads}\n")
        for method, order, arr in results:
            f.write(
                f"  {method} (order={order}): "
                f"{arr.mean():.1f} +/- {arr.std():.1f} particles/sec\n"
            )

    print(f"Benchmark data written to {args.output}")
