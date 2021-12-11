from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

from .base import ConvexBody


class Polytope(ConvexBody):
    """A polytope pol(A, b) is a convex body defined by a collection of linear inequalities. More precisely,
    given a m x d matrix "A" and a m-dimensional vector "b", pol(A, b) corresponds to the set of all data
    points "x" satisfying:

                A[i]^t x <= b[i], for all 1 <= i <= m

    or, more compactly, Ax <= b.

    Parameters
    ----------
    A : ArrayLike
        The m x d matrix of linear inequalities
    b : ArrayLike
        The m-dimensional vector bounding each linear inequalities

    Raises
    ------
    ValueError
        If A and b have incompatible dimensions
    """

    def __init__(self, A: ArrayLike, b: ArrayLike) -> None:
        A = np.asarray(A, dtype='float')
        b = np.asarray(b, dtype='float')

        if A.ndim != 2:
            raise ValueError(f"Array 'A' must be two-dimensional, but ndim={A.ndim}")

        if b.ndim != 1:
            raise ValueError(f"Array 'b' must be one-dimensional, but ndim={b.ndim}")

        if A.shape[0] != b.shape[0]:
            raise ValueError(f"'A' and 'b' have incompatible dimensions: {A.shape[0]} != {b.shape[0]}")

        self._A = A
        self._b = b

    @property
    def A(self) -> np.ndarray:
        return self._A

    @property
    def b(self) -> np.ndarray:
        return self._b

    @property
    def dim(self) -> int:
        return self.A.shape[1]

    def is_inside(self, x: np.ndarray) -> bool:
        return (self.A.dot(x) <= self.b).all()

    def compute_intersection_extremes(self, x: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
        thresholds = self.__compute_thresholds(x, v)

        lower = np.where(thresholds < 0, thresholds, -np.inf).max()
        upper = np.where(thresholds > 0, thresholds, +np.inf).min()

        return lower, upper

    def compute_boundary_reflection(self, x: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, float]:
        thresholds = self.__compute_thresholds(x, v)
        thresholds = np.where(thresholds > 0, thresholds, np.inf)

        idx = thresholds.argmin()
        distance = thresholds[idx]
        internal_normal = -self.A[idx]
        internal_normal /= np.linalg.norm(internal_normal)

        return internal_normal, distance

    def __compute_thresholds(self, x, v) -> np.ndarray:
        return (self.b - self.A.dot(x)) / self.A.dot(v)
