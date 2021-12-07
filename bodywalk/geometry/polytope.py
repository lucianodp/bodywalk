from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

from .base import ConvexBody


class Polytope(ConvexBody):
    """Convex body defined by the intersection of multiple linear equations.
    More precisely, a data point x belongs to a polytope pol(A, b) if:

                        A x <= b

    where "A" is a m x d matrix and "b" a m-dimensional vector.

    Parameters
    ----------
    A : ArrayLike
        The m x d matrix above
    b : ArrayLike
        The m-dimensional vector above

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

    def is_inside(self, x: np.ndarray) -> bool:
        return (self.A.dot(x) <= self.b).all()

    def compute_intersection_extremes(self, x: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
        thresholds = (self.b - self.A.dot(x)) / self.A.dot(v)

        lower = np.where(thresholds < 0, thresholds, -np.inf).max()
        upper = np.where(thresholds > 0, thresholds, +np.inf).min()

        return lower, upper
