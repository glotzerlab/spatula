#!/usr/bin/env python
# Copyright (c) 2021-2025 The Regents of the University of Michigan
# Part of spatula, released under the BSD 3-Clause License.

"""A microbenchmark for PGOP calculation."""

import argparse
import time
import warnings

import coxeter
import freud
import numpy as np
from scipy.spatial.transform import Rotation

import spatula

warnings.filterwarnings(action="ignore")
spatula.util.set_num_threads(1)

SHAPE = coxeter.families.PlatonicFamily.get_shape("Icosahedron")


def run_benchmark(symmetry, n_orientations=50, n_repeats=10, mode="full"):
    """Run the benchmark for a given symmetry.

    Parameters
    ----------
    symmetry : str
        The symmetry to test.
    n_orientations : int, optional
        The number of random orientations to sample. Defaults to 100.
    n_repeats : int, optional
        The number of times to repeat the benchmark. Defaults to 10.
    mode: {"boo", "full"}, optional
        The mode for the PGOP compute. Defaults to "full".

    """
    print(f"--- Benchmarking {symmetry} symmetry ---")

    points = SHAPE.vertices

    # Determine a radius for neighbor finding that includes all vertices
    r_max = 2 * np.max(np.linalg.norm(points, axis=1)) + 0.1

    # Create the spatula optimizer
    # optimizer = spatula.optimize.Union.with_step_gradient_descent(
    # optimizer = spatula.optimize.Mesh.from_grid(n_axes=60, n_angles=10)
    optimizer = spatula.optimize.Mesh.from_lattice(600)
    #     max_iter=500,
    #     learning_rate=0.01,
    # )

    # Create the PGOP object
    pgop = spatula.PGOP(symmetries=[symmetry], optimizer=optimizer, mode=mode)

    # Generate random rotations outside the benchmark loop
    random_rotations = Rotation.random(n_orientations, random_state=42)

    # Store timings and PGOP values
    timings = []
    all_pgop_values = []

    for _ in range(n_repeats):
        pgop_values_one_run = []
        start_time = time.perf_counter()

        for rot in random_rotations:
            rotated_points = rot.apply(points)
            box = freud.box.Box.cube(10.0)
            system = (box, rotated_points)
            pgop.compute(
                system,
                sigmas=0.075 if pgop.mode != "boo" else 177.77,
                neighbors={"r_max": r_max, "exclude_ii": True},
                query_points=[[0, 0, 0]],
            )
            pgop_values_one_run.append(pgop.order[0, 0])

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        timings.append(elapsed_time)
        all_pgop_values.extend(pgop_values_one_run)

    # Timing analysis
    timings = np.array(timings)
    mean_time_total = np.mean(timings)
    std_time_total = np.std(timings)
    mean_time_per_q = mean_time_total / n_orientations
    std_time_per_q = std_time_total / n_orientations

    pgop_values = np.array(all_pgop_values)
    mean_pgop = np.mean(pgop_values)
    std_pgop = np.std(pgop_values)

    print(f"  PGOP: {mean_pgop:.8f} ± {std_pgop:.8f} (mean ± std. dev.)")
    print(
        f"  Time: {mean_time_per_q * 1000:.2f} μs ± "
        f"{std_time_per_q * 1000:.2f} per trial"
        f"(mean ± std. dev. of {n_repeats} runs, {n_orientations} orientations each)\n"
    )


def main():
    """Parse arguments and run the benchmark."""
    symmetries_list = ["C2", "D5d", "T", "Ih"]
    parser = argparse.ArgumentParser(
        description="Run microbenchmarks for spatula PGOP calculations."
    )
    parser.add_argument(
        "--symmetries",
        nargs="+",
        choices=symmetries_list,
        default=symmetries_list,
        help=f"One or more symmetries to benchmark. Defaults to all: {symmetries_list}",
    )
    parser.add_argument(
        "--mode",
        choices=["boo", "full"],
        default="full",
        help="Symmetry mode. Default is 'full'.",
    )
    args = parser.parse_args()

    for symmetry in args.symmetries:
        run_benchmark(symmetry, mode=args.mode)


if __name__ == "__main__":
    main()
