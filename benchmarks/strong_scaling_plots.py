# Copyright (c) 2021-2026 The Regents of the University of Michigan
# Part of spatula, released under the BSD 3-Clause License.

r"""Plot strong scaling benchmark data for PGOP vs MSM runtime.

Run:
```bash
python benchmarks/strong_scaling_plots.py \
   --data-file1 m1_data.txt --data-file2 anvil_data.txt
"""

# ruff: noqa: D103, B023, N806
import argparse
import re
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def parse_data_string(data: str) -> tuple[int, list[tuple[str, int, np.ndarray]]]:
    """Parse the DATA string into results format.

    Returns:
        N_PARTICLES and list of (method, n_threads, times_array) tuples

    """
    lines = data.strip().split("\n")
    n_particles = int(lines[0].split("=")[1])
    results = []

    current_threads = None
    for line in lines[1:]:
        if line.startswith("threads="):
            current_threads = int(line.split("=")[1])
        elif (
            line.strip().startswith("msm:")
            or line.strip().startswith("pgop:")
            or line.strip().startswith("pgop_60:")
            or line.strip().startswith("pgop_none:")
        ):
            match = re.match(
                r"\s*(msm|pgop|pgop_60|pgop_none):\s+([\d.]+)ms\s+\+/-\s+([\d.]+)ms\s+\(([\d.]+)ms/particle\)",
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


def parse_data_file(filepath: Path) -> tuple[int, list[tuple[str, int, np.ndarray]]]:
    """Parse a benchmark data file into results format.

    Returns:
        N_PARTICLES and list of (method, n_threads, times_array) tuples

    """
    with Path.open(filepath) as f:
        return parse_data_string(f.read())


def plot_subplot(ax, results, n_particles, title: str, show_ideal: bool = True):
    """Plot a single subplot with benchmark data."""
    L = 6  # MSM l value

    COLORS = {
        "msm": "#2980B9",
        "pgop": "#C7196A",
        "pgop_60": "#DB7BA3",
        "pgop_none": "#EAB3CA",
    }
    MARKER = {
        "msm": "H",
        "pgop": "o",
        "pgop_60": "s",
        "pgop_none": "^",
    }
    INFO = {
        "msm": f"MSM$_{L}$",
        "pgop": r"PGOP $\mathrm{O_h}$ (600 mesh)",
        "pgop_60": r"PGOP $\mathrm{O_h}$ (60 mesh)",
        "pgop_none": r"PGOP $\mathrm{O_h}$ (no SO(3) opt)",
    }

    all_threads = sorted({r[1] for r in results})

    for method in ["msm", "pgop", "pgop_60", "pgop_none"]:
        data = [(r[1], r[2].mean(), r[2].std()) for r in results if r[0] == method]
        if not data:
            continue
        threads = [d[0] for d in data]
        ms_per_particle = [d[1] / n_particles for d in data]
        std_ms_per_particle = [d[2] / n_particles for d in data]
        means = [1000 / x for x in ms_per_particle]
        stds = [(y**2) * sx / 1000 for y, sx in zip(means, std_ms_per_particle)]

        ax.errorbar(
            threads,
            means,
            yerr=stds,
            label=INFO[method],
            color=COLORS[method],
            marker=MARKER[method],
            capsize=5,
            linewidth=2,
            markersize=5,
        )

        # Plot ideal scaling (1/nthreads) for msm and pgop
        if show_ideal and (method == "pgop" or method == "msm"):
            single_thread_data = [d for d in data if d[0] == 1]
            if single_thread_data:
                baseline_ms_per_particle = single_thread_data[0][1] / n_particles
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

    ax.set_xlabel("Threads")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim((1e2, 1e6))
    ax.set_title(title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file1",
        type=Path,
        required=True,
        help="First benchmark data file",
    )
    parser.add_argument(
        "--data-file2",
        type=Path,
        required=True,
        help="Second benchmark data file",
    )
    parser.add_argument(
        "--label1",
        type=str,
        default=None,
        help="Label for first data file (default: filename)",
    )
    parser.add_argument(
        "--label2",
        type=str,
        default=None,
        help="Label for second data file (default: filename)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Figure_strongscale_combined_log.pdf"),
        help="Output plot file",
    )
    args = parser.parse_args()

    # Parse both data sets
    n_particles1, results1 = parse_data_file(args.data_file1)
    n_particles2, results2 = parse_data_file(args.data_file2)

    # Use provided labels or default to filenames
    label1 = args.label1 or args.data_file1.stem
    label2 = args.label2 or args.data_file2.stem

    # Calculate width ratio based on max thread counts
    max_threads1 = max(r[1] for r in results1)
    max_threads2 = max(r[1] for r in results2)

    # Create side-by-side plot with proportional widths (smaller scaled for readability)
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(12, 5))
    width_ratio = max(max_threads1 / max_threads2, max_threads2 / max_threads1)
    if max_threads1 >= max_threads2:
        gs = fig.add_gridspec(1, 2, width_ratios=[max_threads1, max_threads2 * 4])
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[max_threads1 * 4, max_threads2])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axes[0].set_ylabel("Particles per Second")

    # Plot first data set (left)
    plot_subplot(
        axes[0],
        results1,
        n_particles1,
        f"{label1} ({n_particles1:,} particles)",
    )

    # Plot second data set (right)
    plot_subplot(
        axes[1],
        results2,
        n_particles2,
        f"{label2} ({n_particles2:,} particles)",
    )

    # Adjust x-axis for smaller thread range if needed
    smaller_max = min(max_threads1, max_threads2)
    if smaller_max <= 8:
        axes[1].set_yticklabels([])
        axes[1].set_xlim(-1, smaller_max + 1)
        axes[1].set_xticks([0, smaller_max // 2, smaller_max])

    # Create a single shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    # Add ideal scaling entry
    from matplotlib.lines import Line2D

    handles.append(
        Line2D(
            [0],
            [0],
            linestyle="--",
            color="gray",
            alpha=0.5,
            linewidth=1.5,
            label="Ideal Scaling",
        )
    )
    labels.append("Ideal Scaling")

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(args.output, dpi=450, bbox_inches="tight")
    print(f"Plot saved to {args.output}")

    # Print summary tables
    for name, n_particles, results in [
        (label1, n_particles1, results1),
        (label2, n_particles2, results2),
    ]:
        print(f"\n{name} - Particles per second:")
        for method in ["msm", "pgop", "pgop_60", "pgop_none"]:
            data = [(r[1], r[2].mean(), r[2].std()) for r in results if r[0] == method]
            if not data:
                continue
            print(f"  {method.upper()}:")
            for n_threads, mean_ms, std_ms in sorted(data, key=lambda x: -x[0]):
                ms_per_particle = mean_ms / n_particles
                std_ms_per_particle = std_ms / n_particles
                particles_per_sec = 1000 / ms_per_particle
                std_particles_per_sec = (
                    (particles_per_sec**2) * std_ms_per_particle / 1000
                )
                print(
                    f"    threads={n_threads}: "
                    f"{particles_per_sec:.1f} +/- {std_particles_per_sec:.1f}"
                )

    # Print parallel efficiency for PGOP (600 mesh)
    print("\n" + "=" * 60)
    print("Parallel Efficiency for PGOP (600 mesh):")
    print("Efficiency = (Speedup / threads) * 100%")
    print("=" * 60)
    for name, n_particles, results in [
        (label1, n_particles1, results1),
        (label2, n_particles2, results2),
    ]:
        pgop_data = [(r[1], r[2].mean(), r[2].std()) for r in results if r[0] == "pgop"]
        if not pgop_data:
            continue

        # Get single-thread baseline
        baseline = next((d for d in pgop_data if d[0] == 1), None)
        if not baseline:
            continue
        baseline_pps = 1000 / (baseline[1] / n_particles)

        print(f"\n{name}:")
        print(f"  {'Threads':<10} {'Speedup':<12} {'Efficiency':<12}")
        print(f"  {'-' * 10} {'-' * 12} {'-' * 12}")
        for n_threads, mean_ms, _ in sorted(pgop_data, key=lambda x: -x[0]):
            pps = 1000 / (mean_ms / n_particles)
            speedup = pps / baseline_pps
            efficiency = (speedup / n_threads) * 100
            print(f"  {n_threads:<10} {speedup:<12.2f}x {efficiency:<11.1f}%")
