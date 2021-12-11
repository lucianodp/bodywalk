from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class ConvexBody(ABC):
    """Interface representing a general convex body in the euclidean space, exposing the
    necessary functionality for running all the sampling algorithms supported by this
    library (Ball Walk, Hit-and-Run, ...).

    We also note that each sampling algorithm may only require a small subset of methods
    to be implemented; for example, to run the Ball Walk sampler it is enough to implement
    the is_inside() method. Please refer to the documentation of each sampling function
    for a list of all methods it needs implemented.
    """

    @property
    def dim(self) -> int:
        """
        Returns
        -------
        int
            Return the dimensionality of the underlying euclidean space
        """
        raise NotImplementedError("'dim' property must be implemented to run the Billiard Walk sampler.")

    def is_inside(self, x: np.ndarray) -> bool:
        """Check if a data point is inside this convex body (self).

        This method must be implemented to run the Ball Walk sampling algorithm.

        Parameters
        ----------
        x : np.ndarray
            Data point to verify membership

        Returns
        -------
        bool
            True if x is inside this convex body (self); False otherwise
        """
        raise NotImplementedError("'is_inside' method must be implemented to run the Ball Walk sampler.")

    def compute_intersection_extremes(self, x: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
        """Given an interior point "x" and a direction vector "v", this method computes the
        intersection between the straight line {x + t * v} and this convex body (self). More precisely,
        this method returns a pair of numbers (L, U) satisfying:

                "x + t * v" is inside this convex body (self)  <=>  L <= t <= U.

        This method must be implemented to run the Hit-and-Run sampling algorithm.

        Parameters
        ----------
        x : np.ndarray
            The line's starting point. It is guaranteed to be inside the convex body (self)
        v : np.ndarray
            The line's direction vector. It is NOT guaranteed to be normalized.

        Returns
        -------
        Tuple[float, float]
            The pair (L, U) corresponding to the extremes of the line segment. L must be smaller than U.
        """
        raise NotImplementedError("'compute_intersection_extremes' method must be implemented to run the Hit-and-Run sampler.")

    def compute_boundary_reflection(self, x: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, float]:
        """Given an interior point "x" and a direction vector "v", this method computes where the ray
        {x + tv, t > 0} hits the boundary of this convex body. More precisely, this method returns two
        quantities describing the hit point:

            1) Internal normal: a unit-vector perpendicular to the boundary at the hit point,
                                and pointing inwards to the convex body.

            2) Distance: the distance between "x" and the hit point

        This method must be implemented to run the Billiard Walk sampling algorithm.

        Parameters
        ----------
        x : np.ndarray
            The ray's starting point. It is guaranteed to be inside the convex body (self).
        v : np.ndarray
            The ray's direction vector. It is guaranteed to have unit norm (i.e. ||v|| = 1)

        Returns
        -------
        Tuple[np.ndarray, float]
            The internal normal vector at the hit point, and the distance to the hit point
        """
        raise NotImplementedError("'compute_boundary_reflection' method must be implemented to run the Billiard Walk sampler.")
