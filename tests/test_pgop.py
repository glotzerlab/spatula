# Copyright (c) 2021-2025 The Regents of the University of Michigan
# Part of spatula, released under the BSD 3-Clause License.


import json

import coxeter
import freud
import numpy as np
import pytest
import scipy.spatial

import spatula

RTOL = 1e-4

N_DICT = {
    3: "Triangular",
    4: "Square",
    5: "Pentagonal",
    6: "Hexagonal",
    7: "Heptagonal",
    8: "Octagonal",
    9: "Nonagonal",
    10: "Decagonal",
}


PGOP_DICT = {}


spatula.util.set_num_threads(1)


def get_pyramid(n: int) -> np.ndarray:
    base = coxeter.families.RegularNGonFamily().get_shape(n).vertices
    # Need to offset polygon to make dihedral order low for testing.
    return (
        f"Pyramid({n})",
        np.concatenate((base - np.array([0.0, 0.0, -0.5]), [[0.0, 0.0, 3.0]])),
    )


def get_bipyramid(n: int) -> np.ndarray:
    base = coxeter.families.RegularNGonFamily().get_shape(n).vertices
    return (
        f"Bipyramid({n})",
        np.concatenate((base, [[0.0, 0.0, 3.0], [0.0, 0.0, -3.0]])),
    )


def parse_shape_values(
    a: str | tuple[str, np.ndarray],
) -> tuple[str, np.ndarray]:
    if isinstance(a, (tuple, list)):
        return a
    family, shape = a.split(".")
    if shape.isdigit():
        shape = int(shape)

    vertices = getattr(coxeter.families, family).get_shape(shape).vertices
    return shape, vertices


def _id_func(value):
    if isinstance(value, str):
        return value
    elif isinstance(value, int):
        return f"NGon{value}"
    return ""


# The vertices found in this dictionary for all these shapes were taken from
# http://dmccooey.com/polyhedra/Simplest.html
# molecules are taken from https://symotter.org/gallery
with open("tests/data/shape_vertices.json") as f:  # noqa: PTH123
    SHAPE_SYMMETRIES = json.load(f)

SHAPE_SYMMETRIES.update({f"C{n}v": [get_pyramid(n)] for n in range(4, 8)})
SHAPE_SYMMETRIES.update(
    {
        f"D{n}h": [get_bipyramid(n), f"PrismAntiprismFamily.{N_DICT[n]} Prism"]
        for n in range(3, 7)
    }
)

CUTOFF = 0.99

RNG = np.random.default_rng(seed=42)

METHODS_DICT = {}

# n_axes must be at least 50 for Dnh to work correctly. Further increases bring Dnd
# close to one as well.
OPTIMIZER = spatula.optimize.Union.with_step_gradient_descent(
    spatula.optimize.Mesh.from_grid(), max_iter=500, learning_rate=0.01
)


def get_shape_sys_nlist(vertices):
    """Get a neighbor list of a shape.

    The neighbor list has a single point with all vertices as neighbors.
    """
    l = 100
    box = freud.Box.cube(l)
    system = (box, vertices)
    neighbor_query = freud.locality.AABBQuery.from_system(system)
    query_point = np.zeros((1, 3))
    nlist = neighbor_query.query(
        query_point, {"mode": "ball", "r_max": l * 0.4999}
    ).toNeighborList()
    return system, nlist


def make_compute_object(symmetries, optimizer, optype):
    if optype == "boosop":
        return spatula.BOOSOP("fisher", symmetries, optimizer)
    elif optype == "full":
        return spatula.PGOP(symmetries, optimizer)
    elif optype == "boo":
        return spatula.PGOP(symmetries, optimizer, mode="boo")
    else:
        raise ValueError(f"Invalid optype {optype}")


def make_method(symmetries, optimizer, optype):
    if isinstance(symmetries, str):
        symmetry = symmetries[0]
    else:
        return make_compute_object(symmetries, optimizer, optype)
    if symmetry not in METHODS_DICT:
        METHODS_DICT[symmetry] = {}
    if optype not in METHODS_DICT[symmetry]:
        METHODS_DICT[symmetry][optype] = {}
    if optimizer.__hash__() not in METHODS_DICT[symmetry][optype]:
        METHODS_DICT[symmetry][optype][optimizer.__hash__()] = make_compute_object(
            symmetries, optimizer, optype
        )
    return METHODS_DICT[symmetry][optype][optimizer.__hash__()]


def generate_quaternions(n=1):
    """Generate `n` random quaternions]."""
    rotations = [scipy.spatial.transform.Rotation([1, 0, 0, 0]).as_quat()]
    for _ in range(n):
        rotations.append(
            scipy.spatial.transform.Rotation.random(random_state=RNG).as_quat()
        )
    return rotations


def compute_op_result(
    symmetry, opt, optyp, system, nlist, sigma=None, query_points=None, failed=False
):
    if failed:
        op_compute = make_compute_object(symmetry, opt, optyp)
    else:
        op_compute = make_method(symmetry, opt, optyp)
    if optyp == "boosop":
        op_compute.compute(system, nlist, query_points=query_points)
    elif optyp == "full" or optyp == "boo":
        op_compute.compute(system, sigma, nlist, query_points=query_points)
    return op_compute


def compute_pgop_polyhedron(
    symmetry, vertices, optype, sigma=None, cutoff_operator=">", cutoff_value=CUTOFF
):
    """Determine whether given shape have a specified symmetry.

    Parameters
    ----------
    symmetry: str
        The symmetry to test for.
    vertices: :math:`(N, 3)` numpy.ndarray of floats
        The vertices of the shape
    optype: str
        The type of order parameter to compute. boosop or fpgop or opgop.
    """
    vertices = np.asarray(vertices)
    system, nlist = get_shape_sys_nlist(vertices)
    op_compute = compute_op_result(
        symmetry, OPTIMIZER, optype, system, nlist, sigma, query_points=np.zeros((1, 3))
    )
    if (
        ">" in cutoff_operator
        and op_compute.order[0] < cutoff_value
        or "<" in cutoff_operator
        and op_compute.order[0] > cutoff_value
    ):
        print(f"Used higher precision, lower precision value\n{op_compute.order[0]}")
        new_optimizer = spatula.optimize.Union.with_step_gradient_descent(
            spatula.optimize.Mesh.from_lattice(n_rotations=10_000)
        )
        op_compute = compute_op_result(
            symmetry,
            new_optimizer,
            optype,
            system,
            nlist,
            sigma,
            query_points=np.zeros((1, 3)),
            failed=True,
        )
    return op_compute


modedict_types = ["full", "boo", "boosop"]
crystal_systems = ["sc", "fcc", "bcc"]
crystal_sizes = {"sc": 3, "fcc": 2, "bcc": 2}
crystal_cutoffs = {"sc": 1.1, "fcc": 0.9, "bcc": 0.9}
crystals_dict = {
    "sc": freud.data.UnitCell.sc().generate_system(crystal_sizes["sc"]),
    "fcc": freud.data.UnitCell.fcc().generate_system(crystal_sizes["fcc"]),
    "bcc": freud.data.UnitCell.bcc().generate_system(crystal_sizes["bcc"]),
}


def compute_pgop_check_all_order_values(
    system, symmetry, mode, nlist, sigma=None, qp=None, value=1.0, rtol=RTOL
):
    expected_shape = (
        len(system[1]) if qp is None else len(qp),
        len(symmetry),
    )
    op_pg = compute_op_result(symmetry, OPTIMIZER, mode, system, nlist, sigma, qp)
    assert op_pg.order.shape == expected_shape
    if not np.allclose(op_pg.order, value, rtol=rtol):
        print("Used higher precision, lower precision value", op_pg.order)
        new_optimizer = spatula.optimize.Union.with_step_gradient_descent(
            spatula.optimize.Mesh.from_lattice(n_rotations=5_000),
            max_iter=1_000,
            learning_rate=0.01,
        )
        op_pg = compute_op_result(
            symmetry, new_optimizer, mode, system, nlist, sigma, qp, True
        )
    assert op_pg.order.shape == expected_shape
    return op_pg


def compute_pgop_crystal(crystal_type, symmetry, mode, nlist, sigma=None, qp=None):
    system = crystals_dict[crystal_type]
    return compute_pgop_check_all_order_values(system, symmetry, mode, nlist, sigma, qp)


# Define a parameter for different unit cells and corresponding parameters
@pytest.mark.parametrize("crystal_type", crystal_systems)
@pytest.mark.parametrize("mode", modedict_types)
def test_simple_crystals(crystal_type, mode):
    qargs = {"exclude_ii": True, "mode": "ball", "r_max": crystal_cutoffs[crystal_type]}
    op_pg = compute_pgop_crystal(crystal_type, ["Oh"], mode, qargs, None)
    np.testing.assert_allclose(
        op_pg.order, np.ones((len(crystals_dict[crystal_type][1]), 1)), rtol=RTOL
    )


@pytest.mark.parametrize("mode", modedict_types)
def test_qargs_query_pt(mode):
    qargs = {"exclude_ii": True, "mode": "ball", "r_max": crystal_cutoffs["sc"]}
    _, points = crystals_dict["sc"]
    op_pg = compute_pgop_crystal(
        "sc", ["Oh"], mode, qargs, None, qp=np.asarray([points[0]])
    )
    np.testing.assert_allclose(op_pg.order, np.ones((1, 1)), rtol=RTOL)


@pytest.mark.parametrize("mode", modedict_types)
def test_neighbor_list_query_pt(mode):
    box, points = crystals_dict["sc"]
    qargs = {"exclude_ii": True, "mode": "ball", "r_max": crystal_cutoffs["sc"]}
    qp = np.asarray([points[0]])
    neighborlist = (
        freud.locality.AABBQuery(box, points).query(qp, qargs).toNeighborList()
    )
    op_pg = compute_pgop_crystal("sc", ["Oh"], mode, neighborlist, None, qp)
    np.testing.assert_allclose(op_pg.order, np.ones((1, 1)), rtol=RTOL)


@pytest.mark.parametrize("mode", modedict_types)
def test_neighbor_list_only(mode):
    box, points = crystals_dict["sc"]
    qargs = {"exclude_ii": True, "mode": "ball", "r_max": crystal_cutoffs["sc"]}
    neighborlist = (
        freud.locality.AABBQuery(box, points).query(points, qargs).toNeighborList()
    )
    op_pg = compute_pgop_crystal("sc", ["Oh"], mode, neighborlist, None)
    np.testing.assert_allclose(
        op_pg.order, np.ones((len(crystals_dict["sc"][1]), 1)), rtol=RTOL
    )


@pytest.mark.parametrize("mode", ["full", "boo"])
@pytest.mark.parametrize("sigma", [0.2, [0.2] * (3 * 3 * 3)])
def test_sigma_inputs(mode, sigma):
    box, points = crystals_dict["sc"]
    qargs = {"exclude_ii": True, "mode": "ball", "r_max": crystal_cutoffs["sc"]}
    neighborlist = (
        freud.locality.AABBQuery(box, points).query(points, qargs).toNeighborList()
    )
    op_pg = compute_pgop_crystal("sc", ["Oh"], mode, neighborlist, sigma)
    np.testing.assert_allclose(
        op_pg.order, np.ones((len(crystals_dict["sc"][1]), 1)), rtol=RTOL
    )


MODES = ["full", "boo", "boosop"]
SIGMA_VALUES = {
    "full": 0.2,
    "boo": 19.55,
}


@pytest.mark.parametrize("mode", MODES)
def test_bcc_with_multiple_correct_symmetries(mode):
    qargs = {"exclude_ii": True, "mode": "ball", "r_max": crystal_cutoffs["bcc"]}
    correct_symmetries = ["Oh", "D2", "D4"]
    op_pg = compute_pgop_crystal("bcc", correct_symmetries, mode, qargs, None)
    np.testing.assert_allclose(
        op_pg.order, np.ones((len(crystals_dict["bcc"][1]), 3)), rtol=RTOL
    )


@pytest.mark.parametrize("mode", MODES)
def test_bcc_with_multiple_incorrect_symmetries(mode):
    cutoff = 0.8
    box, points = crystals_dict["bcc"]
    qargs = {"exclude_ii": True, "mode": "ball", "r_max": crystal_cutoffs["bcc"]}
    incorrect_symmetries = ["Oh", "D3h"]
    sigs = SIGMA_VALUES[mode] if mode != "boosop" else None
    op_pg = compute_op_result(
        incorrect_symmetries, OPTIMIZER, mode, (box, points), qargs, sigs
    )
    single_column_shape = (len(crystals_dict["bcc"][1]),)
    np.testing.assert_array_less(
        op_pg.order[:, 1], np.full(single_column_shape, cutoff)
    )
    np.testing.assert_allclose(
        op_pg.order[:, 0], np.ones(single_column_shape), rtol=RTOL
    )


def test_bcc_with_multiple_incorrect_symmetries_operator_calc():
    box, points = crystals_dict["bcc"]
    qargs = {"exclude_ii": True, "mode": "ball", "r_max": crystal_cutoffs["bcc"]}
    correct_symmetries = ["Oh", "D3h"]
    # these two contain identity, but PGOP ignores identity and doesn't count it!!!
    len_oh_sym = len(
        spatula.representations.CartesianRepMatrix(correct_symmetries[0]).matrices
    )
    len_d3h_sym = len(
        spatula.representations.CartesianRepMatrix(correct_symmetries[1]).matrices
    )
    op_pg = spatula.PGOP(
        correct_symmetries,
        OPTIMIZER,
        mode="full",
        compute_per_operator_values_for_final_orientation=True,
    )
    op_pg.compute((box, points), None, qargs)

    # PGOP ignores identity and doesn't count it, so you have N less symmetry ops!!!
    assert np.asarray(op_pg.order).shape == (len(points), len_oh_sym + len_d3h_sym)
    np.testing.assert_array_equal(
        op_pg.order.shape, (len(points), len_oh_sym + len_d3h_sym)
    )

    np.testing.assert_allclose(op_pg.order[:, 0:len_oh_sym], 1.0, atol=RTOL)
    assert not np.allclose(op_pg.order[:, len_oh_sym:], 1.0, atol=RTOL)

    np.testing.assert_array_equal(
        op_pg.rotations.shape, (len(points), len_oh_sym + len_d3h_sym, 4)
    )


SYMMETRIES_SUBGROUP_D5D = ["D5d", "S10", "C2h", "C5v", "D5", "C5", "C2", "Ci", "Cs"]
N_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9]
VERTICES_MULTISIM = np.asarray(
    [
        [0.6180339887498949, 0.0, 1.0],
        [0.6180339887498949, 0.0, -1.0],
        [-0.6180339887498949, 0.0, 1.0],
        [-0.6180339887498949, 0.0, -1.0],
        [0.0, 1.0, 0.6180339887498949],
        [0.0, 1.0, -0.6180339887498949],
        [0.0, -1.0, 0.6180339887498949],
        [0.0, -1.0, -0.6180339887498949],
        [1.0, 0.6180339887498949, 0.0],
        [-1.0, -0.6180339887498949, 0.0],
    ]
)


@pytest.mark.parametrize("n, mode", [(n, mode) for n in N_VALUES for mode in MODES])
def test_increasing_number_of_symmetries(n, mode):
    symmetries_to_compute = []
    for sym in SYMMETRIES_SUBGROUP_D5D[:n]:
        symmetries_to_compute.append(sym)
    system, nlist = get_shape_sys_nlist(VERTICES_MULTISIM)
    op = compute_pgop_check_all_order_values(
        system, symmetries_to_compute, mode, nlist, None, qp=np.zeros((1, 3))
    )
    assert len(op.symmetries) == n
    assert len(op.order[0]) == n
    assert len(op.rotations[0]) == n
    assert np.allclose(op.order, 1.0, rtol=RTOL)


# propello tetrahedron vertices
VERTICES_FOR_TESTING = np.asarray(
    [
        [0.5097553324933856, 0.13968058199610653, 1.0],
        [0.5097553324933856, -0.13968058199610653, -1.0],
        [-0.5097553324933856, -0.13968058199610653, 1.0],
        [-0.5097553324933856, 0.13968058199610653, -1.0],
        [1.0, 0.5097553324933856, 0.13968058199610653],
        [1.0, -0.5097553324933856, -0.13968058199610653],
        [-1.0, -0.5097553324933856, 0.13968058199610653],
        [-1.0, 0.5097553324933856, -0.13968058199610653],
        [0.13968058199610653, 1.0, 0.5097553324933856],
        [0.13968058199610653, -1.0, -0.5097553324933856],
        [-0.13968058199610653, -1.0, 0.5097553324933856],
        [-0.13968058199610653, 1.0, -0.5097553324933856],
        [0.6062678708614785, -0.6062678708614785, 0.6062678708614785],
        [0.6062678708614785, 0.6062678708614785, -0.6062678708614785],
        [-0.6062678708614785, 0.6062678708614785, 0.6062678708614785],
        [-0.6062678708614785, -0.6062678708614785, -0.6062678708614785],
    ]
)


@pytest.mark.parametrize("mode", modedict_types)
@pytest.mark.parametrize("symmetries", [["T"], ["T", "Th"]])
def test_orientations(mode, symmetries):
    # random orientation
    rot = scipy.spatial.transform.Rotation.random(random_state=RNG)
    # compute new vertices
    rotated_vertices = rot.apply(VERTICES_FOR_TESTING)
    system, nlist = get_shape_sys_nlist(rotated_vertices)
    op_opt = compute_op_result(
        symmetries, OPTIMIZER, mode, system, nlist, None, np.zeros((1, 3))
    )
    #   op_opt.rotations is something like [w, x, y, z].
    for cxx_q, symmetry, order in zip(op_opt.rotations[0], symmetries, op_opt.order[0]):
        # Reorder it to [x, y, z, w] to use in SciPy
        scipy_q = np.array([cxx_q[1], cxx_q[2], cxx_q[3], cxx_q[0]])
        optimal_rotation = scipy.spatial.transform.Rotation.from_quat(scipy_q)
        re_rotated_vertices = optimal_rotation.apply(rotated_vertices)
        system, nlist = get_shape_sys_nlist(re_rotated_vertices)
        norot = spatula.optimize.NoOptimization()
        op_no_opt = compute_op_result(
            [symmetry], norot, mode, system, nlist, None, np.zeros((1, 3))
        )
        np.testing.assert_allclose(order, op_no_opt.order[0], rtol=RTOL)


OPTIMIZERS_TO_TEST = [
    (
        "Union_descent_random",
        spatula.optimize.Union.with_step_gradient_descent(
            spatula.optimize.RandomSearch(
                max_iter=10_000, seed=RNG.integers(0, 1000000)
            )
        ),
    ),
    (
        "Union_descent_Mesh",
        spatula.optimize.Union.with_step_gradient_descent(
            spatula.optimize.Mesh.from_grid()
        ),
    ),
    ("Descent", spatula.optimize.StepGradientDescent()),
    (
        "Random",
        "RandomSearch",
    ),
    ("Mesh", spatula.optimize.Mesh([[1, 0, 0, 0]])),
    ("Mesh", spatula.optimize.Mesh([[0, 0, 0, 1]])),
    ("NoOptimization", spatula.optimize.NoOptimization()),
]


# parametrize over all optimizers and all modes
@pytest.mark.parametrize(
    "optim_name, optim",
    OPTIMIZERS_TO_TEST,
    ids=[name for name, _ in OPTIMIZERS_TO_TEST],
)
@pytest.mark.parametrize("mode", modedict_types)
def test_optimization_classes(optim_name, optim, mode):
    # Ensure the pure random optimizer has enough samples to pass
    if "Random" in optim_name:
        optim = spatula.optimize.RandomSearch(max_iter=50_000, seed=0)
    system, nlist = get_shape_sys_nlist(VERTICES_FOR_TESTING)
    op = compute_op_result(["T"], optim, mode, system, nlist, None, np.zeros((1, 3)))
    print(op.order)
    assert op.order[0] > CUTOFF


@pytest.mark.pg_first_only
@pytest.mark.parametrize(
    "symmetry, shape, vertices, quaternion, mode",
    (
        (sym, shape, vertices, quat, mode)
        for sym, shapes in SHAPE_SYMMETRIES.items()
        for shape, vertices in map(parse_shape_values, shapes)
        for quat in generate_quaternions()
        for mode in modedict_types
    ),
    ids=_id_func,
)
def test_symmetries_polyhedra(symmetry, shape, vertices, quaternion, mode):
    rotation = scipy.spatial.transform.Rotation.from_quat(quaternion)
    rotated_vertices = rotation.apply(vertices)
    op = compute_pgop_polyhedron(
        symmetry=[symmetry], vertices=rotated_vertices, optype=mode
    )
    assert op.order[0] >= CUTOFF


@pytest.mark.pg_first_only
# Move shape vertices along their bond vector away or towards the center
# and compute bosoop and pgop. Boosop should be still be 1 but pgop should be smaller!
@pytest.mark.parametrize(
    "symmetry, shape, vertices",
    (
        (sym, shape, vertices)
        for sym, shapes in SHAPE_SYMMETRIES.items()
        for shape, vertices in map(parse_shape_values, shapes)
    ),
    ids=_id_func,
)
def test_radially_imperfect_symmetry_polyhedra(symmetry, shape, vertices):
    vertices = np.asarray(vertices)
    # randomly scale the distance of a random set of vertices for a number between 1.01
    # and 2
    scale = RNG.uniform(0.5, 2, len(vertices))
    new_vertices = []
    for point, sc in zip(vertices, scale):
        new_vertices.append(point * sc)
    new_vertices = np.asarray(new_vertices)
    boosop_compute = compute_pgop_polyhedron(
        [symmetry], new_vertices, "boosop", None, ">", CUTOFF
    )
    opgop_compute = compute_pgop_polyhedron(
        [symmetry], new_vertices, "boo", None, ">", CUTOFF
    )
    fpgop_compute = compute_pgop_polyhedron(
        [symmetry], new_vertices, "full", None, ">", CUTOFF
    )
    if symmetry == "C1":
        np.testing.assert_allclose(
            boosop_compute.order[0], fpgop_compute.order[0], rtol=RTOL
        )
        np.testing.assert_allclose(
            boosop_compute.order[0], opgop_compute.order[0], rtol=RTOL
        )
        np.testing.assert_allclose(
            fpgop_compute.order[0], opgop_compute.order[0], rtol=RTOL
        )
    else:
        assert boosop_compute.order[0] > fpgop_compute.order[0]
        assert opgop_compute.order[0] > fpgop_compute.order[0]

    assert np.round(boosop_compute.order[0], 4) >= CUTOFF
    assert np.round(opgop_compute.order[0], 4) >= CUTOFF
    np.testing.assert_array_less(fpgop_compute.order[0], 1.0 + RTOL)
    np.testing.assert_array_less(boosop_compute.order[0], 1.0 + RTOL)
    np.testing.assert_array_less(opgop_compute.order[0], 1.0 + RTOL)


NON_SHAPE_SYMMETRIES = {
    "O": ["PlatonicFamily.Tetrahedron", "PlatonicFamily.Icosahedron"],
    "Oh": ["PlatonicFamily.Tetrahedron", "PlatonicFamily.Icosahedron"],
    "T": [get_pyramid(5), get_bipyramid(5)],
    "I": ["PlatonicFamily.Octahedron"],
    "Ih": ["PlatonicFamily.Octahedron"],
}
# Symmetries are carefully chosen here as due to rotations and partial ordering
# meeting the threshold can be difficult here.
CYCLIC_NON_SYMMETRY = {
    2: 3,
    3: 4,
    4: 5,
    5: 3,
    6: 3,
    7: 4,
    8: 3,
    9: 4,
    10: 4,
    11: 5,
    12: 5,
}
NON_SHAPE_SYMMETRIES.update(
    {f"C{i}": [f"RegularNGonFamily.{CYCLIC_NON_SYMMETRY[i]}"] for i in range(3, 13)}
)
NON_SHAPE_SYMMETRIES.update({f"D{i}": [get_pyramid(i)] for i in range(3, 13)})

CUT_IN = 0.92


@pytest.mark.parametrize(
    "symmetry, shape, vertices, optype",
    (
        (sym, shape, vertices, optype)
        for sym, shapes in NON_SHAPE_SYMMETRIES.items()
        for shape, vertices in map(parse_shape_values, shapes)
        for optype in ["boosop", "full", "boo"]
    ),
    ids=_id_func,
)
def test_no_symmetries(symmetry, shape, vertices, optype):
    op = compute_pgop_polyhedron(
        symmetry=[symmetry],
        vertices=vertices,
        optype=optype,
        sigma=None,
        cutoff_operator="<",
        cutoff_value=CUT_IN,
    )
    assert op.order[0] < CUT_IN
