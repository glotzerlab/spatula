import conftest
import coxeter
import numpy as np
import pytest


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
    if isinstance(a, tuple):
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


# TODO: Add shapes for T_h which does not include tetrahedron
shape_symmetries = {
    "O": ["PlatonicFamily.Octahedron", "PlatonicFamily.Cube"],
    "Oh": ["PlatonicFamily.Octahedron", "PlatonicFamily.Cube"],
    "T": ["PlatonicFamily.Tetrahedron"],
    "I": ["PlatonicFamily.Icosahedron"],
    "Ih": ["PlatonicFamily.Icosahedron"],
}
shape_symmetries.update({"C2": ["RegularNGonFamily.4"]})
shape_symmetries.update({f"C{i}": [get_pyramid(i)] for i in range(3, 13)})
shape_symmetries.update({"D2": ["RegularNGonFamily.4"]})
shape_symmetries.update({f"D{i}": [get_bipyramid(i)] for i in range(3, 13)})

non_shape_symmetries = {
    "O": ["PlatonicFamily.Tetrahedron", "PlatonicFamily.Icosahedron"],
    "Oh": ["PlatonicFamily.Tetrahedron", "PlatonicFamily.Icosahedron"],
    "T": [get_pyramid(5), get_bipyramid(5)],
    "I": ["PlatonicFamily.Octahedron"],
    "Ih": ["PlatonicFamily.Octahedron"],
}
# Symmetries are carefully chosen here as due to rotations and partial ordering
# meeting the threshold can be difficult here.
cyclic_non_symmetry = {
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
non_shape_symmetries.update(
    {
        f"C{i}": [f"RegularNGonFamily.{cyclic_non_symmetry[i]}"]
        for i in range(3, 13)
    }
)
non_shape_symmetries.update({f"D{i}": [get_pyramid(i)] for i in range(3, 13)})
non_shape_symmetries.update(
    {f"D{i}": [get_bipyramid(cyclic_non_symmetry[i])] for i in range(3, 13)}
)


@pytest.mark.parametrize(
    "symmetry, shape, vertices",
    (
        (sym, shape, vertices)
        for sym, shapes in shape_symmetries.items()
        for shape, vertices in map(parse_shape_values, shapes)
    ),
    ids=_id_func,
)
def test_symmetries(symmetry, shape, vertices):
    conftest.check_symmetry(
        symmetry=symmetry, vertices=vertices, threshold=0.98
    )


@pytest.mark.parametrize(
    "symmetry, shape, vertices",
    (
        (sym, shape, vertices)
        for sym, shapes in non_shape_symmetries.items()
        for shape, vertices in map(parse_shape_values, shapes)
    ),
    ids=_id_func,
)
def test_no_symmetries(symmetry, shape, vertices):
    conftest.check_symmetry(
        symmetry=symmetry, vertices=vertices, threshold=0.8, has_symmetry=False
    )
