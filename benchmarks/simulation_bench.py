#!/usr/bin/env python
# Copyright (c) 2021-2026 The Regents of the University of Michigan
# Part of spatula, released under the BSD 3-Clause License.

"""Benchmark script for computing PGOP on GSD files."""

import argparse
import time
import warnings

import gsd.hoomd
import numpy as np

import spatula

warnings.filterwarnings(action="ignore")


def run_benchmark(
    gsd_path: str,
    symmetries: list[str],
    sigma: float,
    r_max: float,
    mode: str,
    frames: int | None,
    num_threads: int,
) -> None:
    """Benchmark PGOP performance on a GSD simulation frame.

    Parameters
    ----------
    gsd_path : str
        Path to the GSD file.
    symmetries : list[str]
        List of point group symmetries to compute.
    sigma : float
        Gaussian width parameter for PGOP.
    r_max : float
        Cutoff distance for neighbor finding.
    mode : str
        PGOP computation mode ("full" or "boo").
    frames : int | None
        Number of frames to process. If None, process all frames.
    num_threads : int
        Number of threads for parallel computation.

    """
    if num_threads:
        spatula.util.set_num_threads(num_threads)
    print(f"Opening GSD file: {gsd_path}")

    optimizer = spatula.optimize.Union.with_step_gradient_descent(
        spatula.optimize.Mesh.from_lattice(600)
    )

    pgop = spatula.PGOP(symmetries=symmetries, optimizer=optimizer, mode=mode)

    total_particles = 0
    total_time = 0.0
    frame_count = 0
    all_orders = []  # Collect all (system) PGOP values across frames

    with gsd.hoomd.open(gsd_path) as traj:
        n_frames = len(traj)
        frames_to_process = min(frames, n_frames) if frames else n_frames

        print(f"Processing {frames_to_process} frame(s) with symmetries: {symmetries}")
        print(f"Parameters: sigma={sigma}, r_max={r_max}, mode={mode}")
        print("-" * 60)

        for i, frame in enumerate(traj[::-1]):
            if frames and i >= frames:
                break

            # Extract system data from frame
            box = frame.configuration.box
            points = frame.particles.position
            n_particles = len(points)

            # Time the computation
            start_time = time.perf_counter()
            pgop.compute(
                (box, points),
                sigmas=sigma,
                neighbors={"r_max": r_max, "exclude_ii": True},
            )
            elapsed = time.perf_counter() - start_time

            total_particles += n_particles
            total_time += elapsed
            frame_count += 1

            # Report per-frame results
            order = pgop.order
            all_orders.append(order)
            print(
                f"Frame {i}: {n_particles} particles, "
                f"time={elapsed * 1000:.2f} ms, "
                f"PGOP mean={np.mean(order, axis=0)}"
            )

    # Summary statistics
    print("-" * 60)
    print("Summary:")
    print(f"  Total frames: {frame_count}")
    print(f"  Total particles: {total_particles}")
    print(f"  Total time: {total_time:.3f} s")
    print(f"  Average time per frame: {total_time / frame_count * 1000:.2f} ms")
    print(f"  Average time per particle: {total_time / total_particles * 1e6:.2f} μs")

    # PGOP statistics per symmetry
    all_orders = np.concatenate(
        all_orders, axis=0
    )  # Shape: (total_particles, n_symmetries)
    for i, sym in enumerate(symmetries):
        sym_values = all_orders[:, i]
        print(
            f"  {sym}: mean={np.mean(sym_values):.4f}, "
            f"min={np.min(sym_values):.4f}, max={np.max(sym_values):.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark PGOP computation on GSD files"
    )
    parser.add_argument("gsd_path", help="Path to the GSD file")
    parser.add_argument(
        "--symmetries",
        nargs="+",
        default=["Oh"],
        help="Point group symmetries to compute (default: Oh)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.075,
        help="Gaussian width parameter (default: 0.075)",
    )
    parser.add_argument(
        "--r-max",
        type=float,
        default=1.64,
        help="Neighbor cutoff distance (default: 1.64)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "boo"],
        default="full",
        help="PGOP computation mode (default: full)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=1,
        help="Number of frames to process (default: 1)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Number of threads for parallel computation (default: 1)",
    )

    args = parser.parse_args()

    run_benchmark(
        gsd_path=args.gsd_path,
        symmetries=args.symmetries,
        sigma=args.sigma,
        r_max=args.r_max,
        mode=args.mode,
        frames=args.frames,
        num_threads=args.num_threads,
    )
