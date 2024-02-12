"""Classes to optimize over SO(3) for `pgop.PGOP`."""
from pathlib import Path

import numpy as np
import rowan
import scipy as sp

from . import _pgop


# Only here for typing/documentation purposes.
class Optimizer:
    """Base class for optimizers."""


class RandomSearch(Optimizer):
    """Optimize by testing :math:`N` random rotations."""

    def __init__(self, max_iter=150, seed=42):
        """Create a RandomSearch optimizer.

        Parameters
        ----------
        max_iter : int, optional
            The number of rotations to try. Defaults to 150.
        seed : int, optional
            The random number seed to use for generating random rotations.
            Defaults to 42.
        """
        self._cpp = _pgop.RandomSearch(max_iter, seed)


class StepGradientDescent(Optimizer):
    r"""Optimize by rounds of 3 1D gradient descents.

    The optimization uses a 3-vector, :math:`\nu` to represent rotations in
    :math:`SO(3)`. The conversion to the axis-angle representation for
    :math:`\nu` is

    .. eq::

        \alpha = \frac{\nu}{||\nu||} \\
        \theta = ||\nu||.

    The representation is continuous in :math:`SO(3)`. `StepGradientDescent`
    performs potentially multiple rounds of 3 1-dimensional gradient descents,
    one for each dimension to find the local minimum. Each smaller optimization
    is terminated when the improvement in objective is less than the provided
    tolerance. The entire optimization ends when between rounds of optimization
    the decrease in objective is less than the provided tolerance.
    """

    def __init__(
        self,
        initial_point=(1.0, 0.0, 0.0, 0.0),
        max_iter=150,
        initial_jump=0.001,
        learning_rate=0.05,
        tol=1e-6,
    ):
        """Create a `StepGradientDescent` object.

        Parameters
        ----------
        optimizer : Optimizer
            The initial optimizer. The best/final point of this optimizer will
            be sent to the `StepGradientDescent` as the initial point.
        max_iter : int, optional
            The maximum number of iterations before stopping optimization.
            Defaults to 150.
        initial_jump : float, optional
            The size of the initial jump in each dimension to get an initial
            gradient. Defaults to 0.001.
        learning_rate : float, optional
            The degree to move along the gradient. Higher values lead to larger
            moves and can result in quicker convergence or failure to converge.
            Defaults to 0.05.
        tol : float, optional
            The value that when the reduction in the object is less than the
            current optimization stops. The entire optimization stops when the
            objective from the last round of 1 dimensional optimizations is
            below ``tol``. Defaults to 1e-6.
        """
        self._cpp = _pgop.StepGradientDescent(
            _pgop.Quaternion(initial_point).to_axis_angle_3D(),
            max_iter,
            initial_jump,
            learning_rate,
            tol,
        )


def _load_sphere_codes():
    """Load and return the stored numerical solutions to the Tammes problem.

    The file stores the solutions of the Tammes problem up to 249 points.

    Returns
    -------
    sphere_codes : list[numpy.ndarray]
        The list of sphere codes from 1 to 249 points.
    """
    fn = Path(__file__).parent / "sphere-codes.npz"
    with np.load(str(fn)) as data:
        return [arr for arr in data.values()]


class Mesh(Optimizer):
    """Optimize by testing provided rotations."""

    _sphere_codes = _load_sphere_codes()

    def __init__(self, points):
        """Create a Mesh optimizer.

        Parameters
        ----------
        points : :math:`(N, 4)` numpy.ndarray of float
            The rotaional quaternions to test.
        """
        self._cpp = _pgop.Mesh([_pgop.Quaternion(p) for p in points])

    @classmethod
    def from_grid(cls, n_axes=40, n_angles=5):
        """Create a Mesh optimizer that tests rotations on a uniform grid.

        The axes are chosen by the numerical solutions to the Tammes problem and
        angles by equadistant rotations according to the Haar measure.

        Parameters
        ----------
        n_axes : int, optional
            The number of axes to rotate about. Defaults to 40.
        n_angles : int, optional
            The number of angles to rotate per axes. Defaults to 5.

        Returns
        -------
        mesh : Mesh
            The optimizer which will test :math:`N_{axes} \cdot N_{angles}`
            points.
        """
        if n_axes < 1 or n_axes > 250:
            raise ValueError("Can only chose [1, 250] for n_axes.")
        axes = cls._sphere_codes[n_axes - 1].reshape((-1, 1, 3))
        points = np.empty((n_angles * n_axes + 1, 4), dtype=float)
        points[0] = np.array([1.0, 0.0, 0.0, 0.0])
        angles = cls._sample_angles(n_angles).reshape((1, -1, 1))
        points[1:] = rowan.from_axis_angle(
            axes[:, None, :], angles[None, :, None]
        ).reshape((-1, 4))
        return cls(points)

    @staticmethod
    def _sample_angles(n_angles):
        """Find n equally spaced angle rotations w.r.t. the Haar measure."""

        def cdf(theta, root):
            ipi = 1 / np.pi
            sinx = np.sin(theta)
            return (
                ipi * (theta - sinx) - root,
                ipi * (1 - np.cos(theta)),
                ipi * sinx,
            )

        roots = np.linspace(0, 1, n_angles + 1)[1:]
        angles = []
        for r in roots:
            opt = sp.optimize.root_scalar(
                cdf,
                args=(r,),
                method="halley",
                bracket=[0, np.pi],
                x0=r * np.pi,
                fprime=True,
                fprime2=True,
            )
            angles.append(opt.root)
        return np.array(angles)


class LocalMonteCarlo(Optimizer):
    """Optimize by a Monte Carlo search.

    Optimization steps are small jumps in SO(3).
    """

    def __init__(
        self,
        initial_point=None,
        kT=1.0,  # noqa: N803
        max_theta=0.017,
        seed=42,
        iterations=200,
    ):
        """Create a LocalMonteCarlo optimizer

        Parameters
        ----------
        initial_point : tuple[float, float, float, float], optional
            The starting point for the optimization. Defaults to
            ``(1.0, 0.0, 0.0, 0.0)``.
        kT : float, optional
            The pseudo-temperature to perform the simulation over. Higher values
            lead to more explorative searches while lower temperature better
            descent local minima. Defaults to 1.0.
        max_theta : float, optional
            The max angular rotation (in radian) per step. Defaults to 0.017
            which is approximately :math:`1^{\circ}`.
        seed : int, optional
            The random seed to generate trial moves. Defaults to 42.
        iterations : int, optional
            The number of iterations to run the optimization before stopping.
            Defaults to 200.
        """
        if initial_point is None:
            initial_point = ((1.0, 0.0, 0.0, 0.0), np.inf)
        self._cpp = _pgop.QMonteCarlo(
            (_pgop.Quaternion(initial_point[0]), initial_point[1]),
            kT,
            max_theta,
            seed,
            iterations,
        )


class Union(Optimizer):
    """Combine an optimization scheme with a specific secondary optimizer.

    The union optimizer uses the best point from the first optimization to start
    the second optimizer.
    """

    @classmethod
    def with_step_gradient_descent(
        cls,
        optimizer,
        max_iter=150,
        initial_jump=0.001,
        learning_rate=0.055,
        tol=1e-6,
    ):
        """Create a Union optimizer with a `StepGradientDescent` second step.

        Arguments are passed through to the constructor of
        `StepGradientDescent`.

        Parameters
        ----------
        optimizer : Optimizer
            The initial optimizer. The best/final point of this optimizer will
            be sent to the `StepGradientDescent` as the initial point.
        max_iter : int, optional
            The maximum number of iterations before stopping optimization.
            Defaults to 150.
        initial_jump : float, optional
            The size of the initial jump in each dimension to get an initial
            gradient. Defaults to 0.001.
        learning_rate : float, optional
            The degree to move along the gradient. Higher values lead to larger
            moves and can result in quicker convergence or failure to converge.
            Defaults to 0.05.
        tol : float, optional
            The value that when the reduction in the object is less than the
            current optimization stops. The entire optimization stops when the
            objective from the last round of 1 dimensional optimizations is
            below ``tol``. Defaults to 1e-6.
        """
        instance = cls()
        instance._cpp = _pgop.Union.with_step_gradient_descent(
            optimizer._cpp, max_iter, initial_jump, learning_rate, tol
        )
        return instance

    @classmethod
    def with_mc(
        cls,
        optimizer,
        kT=1.0,  # noqa: N803
        max_theta=0.017,
        seed=42,
        iterations=200,
    ):
        """Create a Union optimizer with a `LocalMonteCarlo` second step.

        Arguments are passed through to the constructor of `LocalMonteCarlo`.

        Parameters
        ----------
        optimizer : Optimizer
            The initial optimizer. The best/final point of this optimizer will
            be sent to the `StepGradientDescent` as the initial point.
        initial_point : tuple[float, float, float, float], optional
            The starting point for the optimization. Defaults to
            ``(1.0, 0.0, 0.0, 0.0)``.
        kT : float, optional
            The pseudo-temperature to perform the simulation over. Higher values
            lead to more explorative searches while lower temperature better
            descent local minima. Defaults to 1.0.
        max_theta : float, optional
            The max angular rotation (in radian) per step. Defaults to 0.017
            which is approximately :math:`1^{\circ}`.
        seed : int, optional
            The random seed to generate trial moves. Defaults to 42.
        iterations : int, optional
            The number of iterations to run the optimization before stopping.
            Defaults to 200.
        """
        instance = cls()
        instance._cpp = _pgop.Union.with_mc(
            optimizer._cpp, kT, max_theta, seed, iterations
        )
        return instance
