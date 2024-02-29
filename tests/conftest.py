import freud
import numpy as np

import pgop

pgop.util.set_num_threads(1)


def get_shape_sys_nlist(vertices):
    """Get a neighbor list of a shape.

    The neighbor list has a single point with all vertices as neighbors.
    """
    # shape = coxeter.shapes.ConvexPolyhedron(vertices)
    # shape.centroid = (0.0, 0.0, 0.0)
    # vertices = shape.vertices
    query_point_indices = np.zeros(len(vertices), dtype=int)
    point_indices = np.arange(0, len(vertices), dtype=int)
    distances = np.linalg.norm(vertices, axis=1)
    return (
        (freud.Box.cube(2.1 * np.max(distances)), vertices),
        freud.locality.NeighborList.from_arrays(
            1, len(vertices), query_point_indices, point_indices, distances
        ),
    )


def check_symmetry(symmetry, vertices, threshold, has_symmetry=True):
    """Determine whether given shape have a specified symmetry.

    Parameters
    ----------
    symmetry: str
        The symmetry to test for.
    vertices: :math:`(N, 3)` numpy.ndarray of floats
        The vertices of the shape
    threshold: float
        The threshold below which ``check_symmetry`` returns ``False`` for
        ``has_symmetry == True`` and above which ``check_symmetry`` returns
        ``False`` for ``has_symmetry == False``.
    has_symmetry: bool, optional
        Whether to test if the shape has or does not have the symmetry.
    """
    optimizer = pgop.optimize.Union.with_step_gradient_descent(
        pgop.optimize.Mesh.from_grid(n_angles=20, n_axes=5), max_iter=100
    )
    op_compute = pgop.PGOP("fisher", [symmetry], optimizer, kappa=20.0)
    system, nlist = get_shape_sys_nlist(vertices)
    print(f"{vertices=}, {nlist.distances=}")
    op_compute.compute(
        system, nlist, query_points=np.zeros((1, 3)), m=13, max_l=12
    )
    if has_symmetry:
        assert op_compute.pgop[0] >= threshold
        return
    assert op_compute.pgop[0] <= threshold
