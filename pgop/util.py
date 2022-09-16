import numpy as np

PI_2 = np.pi / 2
PI_4 = np.pi / 4


def sph_to_cart(theta, phi):
    x = np.empty(theta.shape + (3,))
    sin_theta = np.sin(theta)
    x[..., 0] = sin_theta * np.cos(phi)
    x[..., 1] = sin_theta * np.sin(phi)
    x[..., 2] = np.cos(theta)
    return x
