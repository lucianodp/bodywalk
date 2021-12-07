from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class ConvexBody(ABC):
    """Abstract class representing a general convex body in the euclidean space, R^n.
    You can define your own convex bodies by directly subclassing ConvexBody and implementing
    any required abstract methods.
    """

    @abstractmethod
    def compute_intersection_extremes(self, x: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
        """Computes the pair of numbers (lower, upper) corresponding to the extremes of line segment resulting of
        the interesection between this convex body (self) and the straight line L = {x + t * v}.
        In other words, "x + t * v" is inside this convex body if, and only if, L <= t <= U.

        Parameters
        ----------
        x : np.ndarray
            A point on the line. It is guaranteed to be *inside* this convex body (self).
        v : np.ndarray
            The line's direction vector

        Returns
        -------
        Tuple[float, float]
            The pair (L, U) corresponding to the extremes of the line segment. L must be smaller than U.
        """