# Copyright (c) 2021-2026 The Regents of the University of Michigan
# Part of spatula, released under the BSD 3-Clause License.

"""Plot benchmark data comparing PGOP runtime across different symmetry groups.

Run:
```bash
python benchmarks/group_size_scaling_plots.py \
    --data-file=benchmarks/data/group_size_scaling.txt
```
"""

# ruff: noqa: D103, B023, N806
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


def get_symmetry_family(symmetry: str) -> str:
    """Get the family of a symmetry group."""
    if symmetry in {"T", "Td", "Th", "O", "Oh", "I", "Ih"}:
        return "polyhedral"
    elif symmetry.startswith("D"):
        return "dihedral"
    elif symmetry.startswith("S"):
        return "improper"
    else:  # C family (Cs, Ch, Ci, Cn, Cnh, Cnv)
        return "cyclic"


# Family colors
FAMILY_COLORS = {
    "polyhedral": "#C7196A",  # magenta
    "dihedral": "#97D434",  # green
    "cyclic": "#9E1AD3",  # purple
    "improper": "#56B6C7",  # cyan
}

# One marker per family
FAMILY_MARKERS = {
    "cyclic": "o",
    "dihedral": "s",
    "improper": "^",
    "polyhedral": "D",
}


def parse_data_string(
    data: str,
) -> tuple[int, list[tuple[str, int, np.ndarray]], list[tuple[str, float, float]]]:
    """Parse the data string into results format.

    Returns:
        N_PARTICLES, list of (symmetry, order, particles_per_sec_array) tuples,
        list of (msm_name, mean, std) tuples for MSM references

    """
    lines = data.strip().split("\n")
    n_particles = int(lines[0].split("=")[1])
    results = []
    msm_results = []  # List of (name, mean, std) for MSM entries

    for line in lines[1:]:
        # Format: "  Symmetry (order=N): XXXXX.X +/- XX.X particles/sec"
        match = re.match(
            r"\s*(\w+)\s+\(order=(\d+)\):\s+([\d.]+)\s+\+/-\s+([\d.]+)\s+particles/sec",
            line,
        )
        if match:
            symmetry = match.group(1)
            order = int(match.group(2))
            mean_pps = float(match.group(3))
            std_pps = float(match.group(4))
            # Generate particles per second array directly
            particles_per_sec_arr = np.random.default_rng(42).normal(
                mean_pps, std_pps, 20
            )
            if symmetry.startswith("msm_"):
                msm_results.append((symmetry, mean_pps, std_pps))
            else:
                results.append((symmetry, order, particles_per_sec_arr))
            continue

        # Format: "  msm_N: XXXXX.X +/- XX.X particles/sec" (without order)
        msm_match = re.match(
            r"\s*(msm_\d+):\s+([\d.]+)\s+\+/-\s+([\d.]+)\s+particles/sec",
            line,
        )
        if msm_match:
            msm_name = msm_match.group(1)
            msm_mean = float(msm_match.group(2))
            msm_std = float(msm_match.group(3))
            msm_results.append((msm_name, msm_mean, msm_std))

    return n_particles, results, msm_results


def parse_data_file(
    filepath: Path,
) -> tuple[int, list[tuple[str, int, np.ndarray]], list[tuple[str, float, float]]]:
    """Parse a benchmark data file into results format.

    Tries the provided path first, then benchmarks/data/ as a fallback.

    Returns:
        N_PARTICLES, list of (symmetry, order, particles_per_sec_array) tuples,
        list of (msm_name, mean, std) tuples for MSM references

    """
    # Try the provided path first
    if filepath.exists():
        with Path.open(filepath) as f:
            return parse_data_string(f.read())

    # Try benchmarks/data/ as a fallback
    fallback = Path("benchmarks/data") / filepath.name
    if fallback.exists():
        with Path.open(fallback) as f:
            return parse_data_string(f.read())

    msg = f"Data file not found: {filepath} or {fallback}"
    raise FileNotFoundError(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        type=Path,
        required=True,
        help="Benchmark data file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/plots/Figure_symmetry.svg"),
        help="Output plot file",
    )
    args = parser.parse_args()

    # Parse data
    n_particles, results, msm_results = parse_data_file(args.data_file)

    # Plotting
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(sharey=True)

    # Prepare data
    plot_data = []
    for symmetry, order, arr in results:
        family = get_symmetry_family(symmetry)
        for val in arr:
            plot_data.append(
                {
                    "order": order,  # numerical
                    "pps": val,
                    "family": family,
                    "symmetry": symmetry,
                }
            )
    df = pd.DataFrame(plot_data)
    unique_orders = sorted(df["order"].unique())
    symmetry_list = list(df["symmetry"].unique())

    # Plot each symmetry group with family marker
    for symmetry in symmetry_list:
        sym_df = df[df["symmetry"] == symmetry]
        family = sym_df["family"].iloc[0]
        color = FAMILY_COLORS[family]
        marker = FAMILY_MARKERS[family]

        ax.scatter(
            sym_df["order"],  # numerical x-values
            sym_df["pps"],
            c=color,
            marker=marker,
            s=30,
            alpha=0.7,
        )

    # MSM reference lines with different linestyles
    MSM_LINESTYLES = ["--", "-.", ":"]  # dashed, dotdash, dotted
    MSM_COLOR = "#2980B9"

    for i, (_, msm_mean, _) in enumerate(msm_results):
        linestyle = MSM_LINESTYLES[i % len(MSM_LINESTYLES)]
        ax.axhline(
            y=msm_mean,
            color=MSM_COLOR,
            linestyle=linestyle,
            linewidth=2,
            alpha=0.8,
        )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker=FAMILY_MARKERS[family],
            color="w",
            markerfacecolor=color,
            markersize=8,
            label=family.capitalize(),
        )
        for family, color in FAMILY_COLORS.items()
    ]
    # Add MSM lines to legend
    for i, (msm_name, _, _) in enumerate(msm_results):
        linestyle = MSM_LINESTYLES[i % len(MSM_LINESTYLES)]
        # Extract the l value from msm_N
        l_val = msm_name.split("_")[1] if "_" in msm_name else "?"
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=MSM_COLOR,
                linestyle=linestyle,
                linewidth=2,
                label=f"MSM$_{l_val}$",
            )
        )
    ax.legend(handles=legend_elements, loc="upper right")

    ax.set_xlabel("Group Order")
    ax.set_xscale("linear")
    ax.set_xlim((0, 50))
    ax.set_ylabel("Particles per second")
    ax.set_yscale("linear")
    ax.set_title(rf"PGOP Performance vs Symmetry Order for {n_particles:,} Particles")

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=450, bbox_inches="tight")
    print(f"Plot saved to {args.output}")

    # Print table
    print("\nPerformance by symmetry:")
    for msm_name, msm_mean, msm_std in msm_results:
        print(f"  {msm_name}: {msm_mean:.1f} +/- {msm_std:.1f} particles/sec")
    results_sorted = sorted(results, key=lambda x: x[1])
    for symmetry, order, arr in results_sorted:
        print(
            f"  {symmetry} (order={order}): "
            f"{arr.mean():.1f} +/- {arr.std():.1f} particles/sec"
        )
