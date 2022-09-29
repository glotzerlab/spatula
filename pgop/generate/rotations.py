import numpy as np
import rowan as rn


class PointGroupRotations:
    # We only store the required quaternions to avoid the 2 to 1 mapping problem
    # with quaternions where the negative of a quaternion is the same rotation
    # as the original quaternion.
    _Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    _Hurwitz = np.array(
        [
            [0.5, 0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, 0.5],
            [-0.5, -0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5, 0.5],
            [-0.5, 0.5, -0.5, 0.5],
            [0.5, -0.5, -0.5, 0.5],
            [-0.5, -0.5, -0.5, 0.5],
        ]
    )
    _Lipschitz = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, -1.0, 1.0],
        ]
    )
    # Remaining members of the isocans
    _isocans = [
        np.array([0.0, 0.5, 0.30901699, 0.80901699]),
        np.array([0.0, -0.5, 0.30901699, 0.80901699]),
        np.array([0.0, 0.5, -0.30901699, 0.80901699]),
        np.array([0.0, -0.5, -0.30901699, 0.80901699]),
        np.array([0.5, 0.30901699, 0.0, 0.80901699]),
        np.array([-0.5, 0.30901699, 0.0, 0.80901699]),
        np.array([0.5, -0.30901699, 0.0, 0.80901699]),
        np.array([-0.5, -0.30901699, 0.0, 0.80901699]),
        np.array([0.5, 0.80901699, 0.30901699, 0.0]),
        np.array([-0.5, 0.80901699, 0.30901699, 0.0]),
        np.array([0.5, -0.80901699, 0.30901699, 0.0]),
        np.array([-0.5, -0.80901699, 0.30901699, 0.0]),
        np.array([0.30901699, 0.0, 0.5, 0.80901699]),
        np.array([-0.30901699, 0.0, 0.5, 0.80901699]),
        np.array([0.30901699, 0.0, -0.5, 0.80901699]),
        np.array([-0.30901699, 0.0, -0.5, 0.80901699]),
        np.array([0.30901699, 0.80901699, 0.0, 0.5]),
        np.array([-0.30901699, 0.80901699, 0.0, 0.5]),
        np.array([0.30901699, -0.80901699, 0.0, 0.5]),
        np.array([-0.30901699, -0.80901699, 0.0, 0.5]),
        np.array([0.80901699, 0.30901699, 0.5, 0.0]),
        np.array([-0.80901699, 0.30901699, 0.5, 0.0]),
        np.array([0.80901699, -0.30901699, 0.5, 0.0]),
        np.array([-0.80901699, -0.30901699, 0.5, 0.0]),
        np.array([0.80901699, 0.0, 0.30901699, 0.5]),
        np.array([-0.80901699, 0.0, 0.30901699, 0.5]),
        np.array([0.80901699, 0.0, -0.30901699, 0.5]),
        np.array([-0.80901699, 0.0, -0.30901699, 0.5]),
        np.array([0.5, 0.0, 0.80901699, 0.30901699]),
        np.array([-0.5, 0.0, 0.80901699, 0.30901699]),
        np.array([0.5, 0.0, -0.80901699, 0.30901699]),
        np.array([-0.5, 0.0, -0.80901699, 0.30901699]),
        np.array([0.30901699, 0.5, 0.80901699, 0.0]),
        np.array([-0.30901699, 0.5, 0.80901699, 0.0]),
        np.array([0.30901699, -0.5, 0.80901699, 0.0]),
        np.array([-0.30901699, -0.5, 0.80901699, 0.0]),
        np.array([0.80901699, 0.5, 0.0, 0.30901699]),
        np.array([-0.80901699, 0.5, 0.0, 0.30901699]),
        np.array([0.80901699, -0.5, 0.0, 0.30901699]),
        np.array([-0.80901699, -0.5, 0.0, 0.30901699]),
        np.array([0.0, 0.30901699, 0.80901699, 0.5]),
        np.array([0.0, -0.30901699, 0.80901699, 0.5]),
        np.array([0.0, 0.30901699, -0.80901699, 0.5]),
        np.array([0.0, -0.30901699, -0.80901699, 0.5]),
        np.array([0.0, 0.80901699, 0.5, 0.30901699]),
        np.array([0.0, -0.80901699, 0.5, 0.30901699]),
        np.array([0.0, 0.80901699, -0.5, 0.30901699]),
        np.array([0.0, -0.80901699, -0.5, 0.30901699]),
    ]
    _cyclic_axis = np.array([0.0, 0.0, 1.0])
    _dihedral_axis = np.array([1.0, 0.0, 0.0])

    def cyclic(self, order):
        return self.cyclic_quaternions(order, self._cyclic_axis)

    def dihedral(self, order):
        Cn = self.cyclic(order)
        C2 = self.cyclic_quaternions(2, self._dihedral_axis)
        return self.prod(Cn, C2)

    def tetrahedral(self):
        # Create the 8 quaternions in group Q
        return np.concatenate((self._Q, self._Hurwitz), axis=0)

    def octahedral(self):
        return np.concatenate((self._Q, self._Hurwitz, self._Lipschitz), axis=0)

    def icosahedral(self):
        return np.concatenate((self._Q, self._Hurwitz, self._isocans), axis=0)

    @staticmethod
    def prod(a, b):
        return rn.multiply(a[None, ...], b[:, None, :]).reshape((-1, 4))

    @staticmethod
    def cyclic_quaternions(order, axis=(0.0, 0.0, 1.0)):
        q = rn.from_axis_angle(axis, np.linspace(0, 2 * np.pi, order + 1)[:-1])
        q[0] = np.array([1.0, 0.0, 0.0, 0.0])
        return q

    def __getitem__(self, schonflies_symbol):
        if schonflies_symbol == "I":
            return self.icosahedral()
        if schonflies_symbol == "O":
            return self.octahedral()
        if schonflies_symbol == "T":
            return self.tetrahedral()
        if schonflies_symbol.startswith("C"):
            return self.cyclic(int(schonflies_symbol[1:]))
        if schonflies_symbol.startswith("D"):
            return self.cyclic(int(schonflies_symbol[1:]))
        else:
            raise KeyError(
                f"Point group {schonflies_symbol} is not currently supported."
            )
