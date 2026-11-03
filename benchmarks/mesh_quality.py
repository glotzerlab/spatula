#!/usr/bin/env python
# Copyright (c) 2021-2026 The Regents of the University of Michigan
# Part of spatula, released under the BSD 3-Clause License.

"""Audit the quality of SO(3) orientation meshes: from_lattice vs from_grid.

Both ``from_lattice`` variants are audited: the identity-anchored Fibonacci
lattice now shipped (node 0 is the identity, node ``i >= 1`` sits at ramp
position ``i + psi / 2``) and the legacy ``s = i + 1/2`` lattice it replaced,
showing that the identity anchor costs essentially nothing in uniformity.

* **Voronoi cell volume** (median with 25th/75th percentile error bars): how
  evenly SO(3) is partitioned. A perfectly uniform mesh gives every rotation a
  cell of volume ``pi**2 / N``, although this is not attainable for arbitrary n.
* **Minimum nearest-neighbor distance** (separation): the smallest distance
  between any two closest mesh points; small values mean wasted points.
* **Maximum nearest-neighbor distance**: the largest distance from a mesh
  point to its closest neighbor, i.e. the biggest hole in the mesh.
"""

import argparse
import contextlib
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rowan
from scipy.spatial import ConvexHull, QhullError, SphericalVoronoi, cKDTree

import spatula.optimize as opt

warnings.filterwarnings("ignore", "Mesh.from_grid is deprecated")

# n_axes for the two from_grid variants plus logspace-ish counts for the lattice.
GRID10_AXES = [5, 15, 40, 75, 130, 250]
GRID30_AXES = [4, 9, 19, 42, 135, 250]
LATTICE_N = sorted(
    {10 * n_axes + 1 for n_axes in GRID10_AXES}
    | {30 * n_axes + 1 for n_axes in GRID30_AXES}
    | {10010, 30000, 99990}
)
SERIES = ("lattice", "lattice_legacy", "grid10", "grid30")
MEASURED_NAMES = ("N", "d_min", "d_max", "p25", "p50", "p75")
N_CELLS_SAMPLED = 65536  # Voronoi cells per mesh for the volume percentiles

PALETTE = {
    "ink": "#030519",
    "mauve": "#71618d",
    "gray": "#6c6575",
    "lattice": "#00c9a4",
    "lattice_legacy": "#4d8075",
    "grid10": "#ff9b7d",
    "grid30": "#c76e96",
}
N_ANGLES = {"grid10": 10, "grid30": 30}
LABELS = {
    "lattice": "from_lattice",
    "lattice_legacy": "from_lattice (legacy, unanchored)",
    "grid10": "from_grid (n_angles=10)",
    "grid30": "from_grid (n_angles=30)",
}
MARKERS = {"lattice": "o", "lattice_legacy": "D", "grid10": "s", "grid30": "^"}
MEASURED_PROPERTIES = {
    "markersize": 5,
    "markeredgecolor": PALETTE["ink"],
    "markeredgewidth": 0.5,
    "linewidth": 1.6,
}


def lattice_quaternions(n):
    """Return the n rotations of ``Mesh.from_lattice(n)`` as [w, x, y, z]."""
    return opt._quaternion_fibonacci_lattice(n)


def legacy_lattice_quaternions(n):
    """Return the pre-identity-anchor from_lattice rotations, as [w, x, y, z].

    spatula's lattice originally placed every node at ramp position
    ``s = i + 1/2`` with no anchored identity; the current from_lattice moved
    node 0 to the identity and the remaining nodes to ``s = i + psi / 2``.
    """
    psi = 1.533751168755204288118041413  # Solution to psi**4 = psi + 4
    s = np.arange(n) + 1 / 2
    t = s / n
    d = 2 * np.pi * s
    r0, r1 = np.sqrt(t), np.sqrt(1 - t)
    alpha, beta = d / np.sqrt(2), d / psi
    return np.column_stack(
        (
            r0 * np.sin(alpha),
            r0 * np.cos(alpha),
            r1 * np.sin(beta),
            r1 * np.cos(beta),
        )
    )


def grid_quaternions(n_axes, n_angles):
    """Return the rotations of ``Mesh.from_grid(n_axes, n_angles)`` as [w, x, y, z].

    ``Mesh`` does not expose its points, so this reproduces the construction
    from_grid performs with the same stored Tammes sphere codes and the same
    Haar-quantile angles; rowan broadcasts the (n_axes, 1, 3) axes against
    the (1, n_angles) angles to form the full axis-angle product.
    """
    axes = opt.Mesh._sphere_codes[n_axes - 1]
    angles = opt.Mesh._sample_angles(n_angles)
    points = np.empty((n_angles * n_axes + 1, 4))
    points[0] = [1.0, 0.0, 0.0, 0.0]
    points[1:] = rowan.from_axis_angle(axes[:, None, :], angles[None, :]).reshape(-1, 4)
    return points


def canonical_rotations(quaternions):
    """Deduplicate rotations, keeping one unit quaternion per rotation."""
    quaternions = rowan.normalize(np.asarray(quaternions, dtype=float))
    first = np.argmax(np.abs(quaternions) > 1e-8, axis=1)
    flip = quaternions[np.arange(len(quaternions)), first] < 0
    quaternions[flip] *= -1
    return np.unique(np.round(quaternions, 9), axis=0)


def nearest_neighbor_angles(quaternions):
    """Min and max nearest-neighbor rotation angles of a mesh, in degrees."""
    doubled = np.vstack((quaternions, -quaternions))
    d = cKDTree(doubled).query(quaternions, k=2)[0][:, 1]
    cos_theta = np.clip(1.0 - d**2 / 2.0, 0.0, 1.0)
    omega = np.degrees(2.0 * np.arccos(cos_theta))
    return omega.min(), omega.max()


def cell_volume_percentiles(quaternions):
    """25th/50th/75th percentiles of the SO(3) Voronoi cell volumes.

    Returns None when no spherical Voronoi diagram exists.
    """
    doubled = np.vstack((quaternions, -quaternions))
    try:
        sv = SphericalVoronoi(doubled)
    except ValueError:
        return None
    rng = np.random.default_rng(42)
    idx = rng.choice(
        len(doubled), size=min(N_CELLS_SAMPLED, len(doubled)), replace=False
    )
    volumes = []
    for i in idx:
        pts = np.vstack((np.zeros(4), sv.vertices[sv.regions[i]]))
        try:
            volumes.append(4.0 * ConvexHull(pts, qhull_options="Q12").volume)
        except QhullError:
            with contextlib.suppress(QhullError):
                volumes.append(4.0 * ConvexHull(pts, qhull_options="QJ").volume)
    if len(volumes) < 0.9 * len(idx):
        return None
    return np.percentile(volumes, [25, 50, 75])


def audit(quaternions):
    """Compute the three analyses for one mesh."""
    quaternions = canonical_rotations(quaternions)
    d_min, d_max = nearest_neighbor_angles(quaternions)
    volumes = cell_volume_percentiles(quaternions)
    return {
        "N": len(quaternions),
        "d_min": d_min,
        "d_max": d_max,
        "p25": np.nan if volumes is None else volumes[0],
        "p50": np.nan if volumes is None else volumes[1],
        "p75": np.nan if volumes is None else volumes[2],
    }


def series_meshes(series):
    """Yield (label, quaternions) for each mesh of a series."""
    if series == "lattice":
        for n in LATTICE_N:
            yield f"from_lattice(n={n})", lattice_quaternions(n)
    elif series == "lattice_legacy":
        for n in LATTICE_N:
            yield f"from_lattice (legacy, n={n})", legacy_lattice_quaternions(n)
    elif series == "grid10":
        for n_axes in GRID10_AXES:
            yield (
                f"from_grid(n_axes={n_axes}, n_angles=10)",
                grid_quaternions(n_axes, 10),
            )
    elif series == "grid30":
        for n_axes in GRID30_AXES:
            yield (
                f"from_grid(n_axes={n_axes}, n_angles=30)",
                grid_quaternions(n_axes, 30),
            )
    else:
        raise ValueError(f"Unknown series {series}.")


def audit_series(series, rows):
    """Audit the not-yet-cached meshes of a series, yielding after each one.

    ``rows`` holds the results computed so far for this series; the meshes it
    already covers are skipped, so interrupted runs resume where they stopped.
    One line is printed per mesh, and the updated ``rows`` are yielded after each mesh
    """
    meshes = list(series_meshes(series))
    for label, quaternions in meshes[len(rows["N"]) :]:
        result = audit(quaternions)
        for name in MEASURED_NAMES:
            rows[name].append(result[name])
        volume = (
            "no Voronoi diagram (rank-deficient mesh)"
            if np.isnan(result["p50"])
            else "cell volume p25/p50/p75 = "
            f"{result['p25']:.4f}/{result['p50']:.4f}/{result['p75']:.4f}"
            f" (ideal {np.pi**2 / result['N']:.4f})"
        )
        print(
            f"{label}: N={result['N']}, neighbors {result['d_min']:.2f}-"
            f"{result['d_max']:.2f} deg, {volume}",
            flush=True,
        )
        yield rows


def load_cache(path):
    """Return the cached audits that match the mesh lists."""
    try:
        with np.load(path) as data:
            if not (
                np.array_equal(data["grid10_axes"], GRID10_AXES)
                and np.array_equal(data["grid30_axes"], GRID30_AXES)
                and np.array_equal(data["lattice_n"], LATTICE_N)
            ):
                return {}
            return {
                series: {name: data[f"{series}_{name}"] for name in MEASURED_NAMES}
                for series in SERIES
                if f"{series}_N" in data
            }
    except (OSError, KeyError):
        return {}


def save_cache(path, data):
    """Write the audits together with the mesh lists that produced them."""
    np.savez(
        path,
        grid10_axes=GRID10_AXES,
        grid30_axes=GRID30_AXES,
        lattice_n=LATTICE_N,
        **{
            f"{series}_{name}": values
            for series, rows in data.items()
            for name, values in rows.items()
        },
    )


def plot(data, output):
    """Draw the three stacked analysis panels."""
    angles = {na: np.degrees(opt.Mesh._sample_angles(na)) for na in (10, 30)}
    theta1 = {series: angles[na][0] for series, na in N_ANGLES.items()}

    rc = {
        "text.color": PALETTE["ink"],
        "axes.labelcolor": PALETTE["ink"],
        "axes.titlecolor": PALETTE["ink"],
        "xtick.color": PALETTE["gray"],
        "ytick.color": PALETTE["gray"],
        "axes.edgecolor": PALETTE["gray"],
    }
    guide_properties = {"color": PALETTE["mauve"], "linestyle": ":", "linewidth": 1}
    n_ref = np.geomspace(50, 110_000, 128)

    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            3, 1, figsize=(7.5, 9.5), sharex=True, constrained_layout=True
        )

        # (a) Voronoi cell volume: median with 25th-75th percentile error bars.
        ax = axes[0]
        for series, rows in data.items():
            ax.errorbar(
                rows["N"],
                rows["p50"],
                yerr=[rows["p50"] - rows["p25"], rows["p75"] - rows["p50"]],
                color=PALETTE[series],
                ecolor=PALETTE[series],
                elinewidth=1.2,
                capsize=2.5,
                marker=MARKERS[series],
                label=LABELS[series],
                linestyle="none",
                **MEASURED_PROPERTIES,
            )
        ax.plot(
            n_ref, np.pi**2 / n_ref, label=r"$\pi^2/N$ (uniform)", **guide_properties
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("Voronoi cell volume [rad$^3$]")
        ax.set_title("SO(3) Voronoi cell volume (median, 25th–75th percentile)")
        ax.legend(loc="lower left", fontsize=8)
        # from_grid runs out of stored Tammes axes at 250.
        for series, n_cap in (("grid10", 2501), ("grid30", 7501)):
            ax.axvline(n_cap, color=PALETTE[series], linewidth=0.75, alpha=0.5)
        ax.text(
            2510,
            ax.get_ylim()[0],
            "n_axes = 250 cap",
            fontsize=7,
            rotation=90,
            va="bottom",
            color=PALETTE["gray"],
        )

        # (b) Minimum nearest-neighbor distance.
        ax = axes[1]
        for series, rows in data.items():
            ax.plot(
                rows["N"],
                rows["d_min"],
                color=PALETTE[series],
                marker=MARKERS[series],
                label=LABELS[series],
                **MEASURED_PROPERTIES,
            )
        ax.plot(
            n_ref,
            np.degrees((np.pi**2 / n_ref) ** (1 / 3)),
            label=r"$(\pi^2/N)^{1/3}$ (uniform spacing)",
            **guide_properties,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("degrees")
        ax.set_title("Minimum nearest-neighbor distance (separation)")
        ax.legend(loc="lower left", fontsize=8)

        # (c) Maximum nearest-neighbor distance. from_grid's value is theta_1
        # at every n_axes, so each of those series is a labeled dashed
        # reference line with the measured points drawn on top of it.
        ax = axes[2]
        for series in ("lattice", "lattice_legacy"):
            rows = data[series]
            ax.plot(
                rows["N"],
                rows["d_max"],
                color=PALETTE[series],
                marker=MARKERS[series],
                label=LABELS[series],
                **MEASURED_PROPERTIES,
            )
        for series in ("grid10", "grid30"):
            rows = data[series]
            ax.axhline(
                theta1[series],
                color=PALETTE[series],
                linestyle="--",
                linewidth=1,
                label=f"{LABELS[series]}: "
                f"$\\theta_1$ = {theta1[series]:.1f}\N{DEGREE SIGN}",
            )
            ax.plot(
                rows["N"],
                rows["d_max"],
                color=PALETTE[series],
                marker=MARKERS[series],
                linestyle="none",
                **MEASURED_PROPERTIES,
            )
        ax.plot(
            n_ref,
            np.degrees((np.pi**2 / n_ref) ** (1 / 3)),
            label=r"$(\pi^2/N)^{1/3}$ (uniform spacing)",
            **guide_properties,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Total number of rotations")
        ax.set_ylabel("degrees")
        ax.set_title("Maximum nearest-neighbor distance (largest hole in the mesh)")
        ax.legend(loc="lower left", fontsize=8)

        fig.savefig(output, format="svg")
        print(f"wrote {output}")


def main():
    """Parse arguments, audit the meshes, and draw the comparison plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="mesh_quality.svg")
    parser.add_argument("--cache", default="mesh_quality.npz")
    args = parser.parse_args()

    data = load_cache(args.cache)
    for series in SERIES:
        rows = {
            name: list(data.get(series, {}).get(name, [])) for name in MEASURED_NAMES
        }
        for updated in audit_series(series, rows):
            data[series] = {
                name: np.asarray(values, dtype=float)
                for name, values in updated.items()
            }
            # Checkpoint after every mesh so interrupted runs can resume.
            save_cache(args.cache, data)
    plot(data, args.output)


if __name__ == "__main__":
    main()
