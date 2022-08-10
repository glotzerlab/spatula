import numpy as np
import scipy as sp
import scipy.special
import scipy.stats


def partial_surface_area(R, delta_theta):
    return 2 * np.pi * R * R * (1 - np.cos(delta_theta))


def central_angle(theta_a, phi_a, theta_b, phi_b):
    """Compute the central angles between sets of points.

    Assumes the first column is the polar (latitude) and the second is the
    azmuthal angle. The polar angle is assumed to go from 0 to pi.
    """
    ta = theta_a - (np.pi / 2)
    tb = theta_b - (np.pi / 2)
    return _central_angle_fast(
        np.sin(ta, dtype=np.float32),
        np.sin(tb, dtype=np.float32),
        np.cos(ta, dtype=np.float32),
        np.cos(tb, dtype=np.float32),
        phi_a,
        phi_b)


def _central_angle_fast(sin_ta, sin_tb, cos_ta, cos_tb, phi_a, phi_b):
    return np.arccos(
        np.multiply(sin_ta, sin_tb, dtype=np.float32)
        + np.multiply(
            np.multiply(cos_ta, cos_tb, dtype=np.float32),
            np.cos(np.abs(np.subtract(phi_a,  phi_b, dtype=np.float32)))
        )
    )


def sph_to_cart(theta, phi):
    x = np.empty(theta.shape + (3,))
    sin_theta = np.sin(theta)
    x[..., 0] = sin_theta * np.cos(phi)
    x[..., 1] = sin_theta * np.sin(phi)
    x[..., 2] = np.cos(theta)
    return x


def project_to_unit_sphere(x):
    return (
        np.arccos(x[:, 2] / np.sqrt((x * x).sum(axis=1))),
        np.arctan2(x[:, 1], x[:, 0])
    )


def delta(x, y):
    return 1 if x == y else 0


# We use 10 as an arbitary number to move deviations from the itertial frame
# away from zero matrix elements.
_fake_mass = 10
_ref_I = np.diag(np.full(3, 0.4 * _fake_mass))


def get_inertial_tensor(dist):
    dist = 4 * dist / np.linalg.norm(dist, axis=1)[:, None]
    dot = np.einsum("ij,ij,kl->ikl", dist, dist, np.eye(3))
    outer = np.einsum("ij,ik->ijk", dist, dist)
    return np.sum(dot - outer, axis=0)


def order_rotation_matrix(ein_values, ein_vectors):
    return ein_vectors[np.argsort(ein_values)]


def normalize_vectors(a):
    return a / np.linalg.norm(a, axis=1)[:, None]


def diagonalize_neighborhood(dist):
    I = get_inertial_tensor(normalize_vectors(dist))
    diag_I, rot_matrix = np.linalg.eig(I)
    return np.einsum("ij,ki->kj", order_rotation_matrix(diag_I, rot_matrix), dist)
