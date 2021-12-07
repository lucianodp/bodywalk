import math
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

from .base import ConvexBody


class Ball(ConvexBody):
    """Class representing a ball in the euclidean space.

    Parameters
        ----------
        center : ArrayLike
            The center of the ball
        radius : float
            The radius of the ball

        Raises
        ------
        ValueError
            If the radius is either 0 or negative
    """

    def __init__(self, center: ArrayLike, radius: float) -> None:
        center = np.asarray(center, dtype='float')

        if radius <= 0:
            raise ValueError(f"radius of ball must be positive, but got {radius}")

        self._center = center
        self._radius = radius

    @property
    def center(self) -> np.ndarray:
        return self._center

    @property
    def radius(self) -> float:
        return self._radius

    def compute_intersection_extremes(self, x: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
        disp = self._center - x

        a = v.dot(v)
        b = v.dot(disp)
        c = disp.dot(disp) - self._radius * self._radius

        return self.__solve_second_degree_equation(a, b, c)

    @staticmethod
    def __solve_second_degree_equation(a: float, b: float, c: float) -> Tuple[float, float]:
        """Finds the roots of the second degree equation ax^2 + 2bx + c = 0"""
        sq_delta = math.sqrt(b * b - a * c)
        return (b - sq_delta) / a, (b + sq_delta) / a