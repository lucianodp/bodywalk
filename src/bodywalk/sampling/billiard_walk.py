from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from .markov import MarkovChain

from ..geometry import ConvexBody


class billiard_walk(MarkovChain):

    def __init__(
        self,
        body: ConvexBody, initial_point: ArrayLike,
        tau: float = 1, max_reflections: Optional[int] = None
    ):
        """Generate a Markov Chain inside a convex body converging to the uniform distribution
        via the Billiard Walk algorithm.

        Parameters
        ----------
        body : ConvexBody
            The convex body to sample from. It requires an implementation of the
            'compute_boundary_reflection' method.
        initial_point : ArrayLike
            The starting point for the Markov chain. It must be inside the convex body.
        tau : float (default = 1)
            Average length of billiard trajectory.
        max_reflections: Optional[int] (default = None)
            Maximum number of reflections allowed per billiard trajectory. If None,
            an heuristic of 10 * n will be used, n being the dimensionality.

        Returns
        -------
        MarkovChain
            A MarkovChain object that generates samples form the convex body via the Billiard Walk strategy.

        Raises
        ------
        ValueError
            If either tau or max_reflections is negative or zero.

        References
        ----------
            [1] B. Polyak, E. Gryazina. Random sampling: Billiard Walk algorithm
                European Journal of Operational Research, 2012
                https://arxiv.org/abs/1211.3932
        """
        if tau <= 0:
            raise ValueError(f"tau must be a positive number, but got {tau}")

        if max_reflections is None:
            max_reflections = 10 * body.dim

        if max_reflections <= 0:
            raise ValueError(f"max_reflections must be a positive integer, but got {max_reflections}")

        super().__init__(body, initial_point)

        self.tau = tau
        self.max_reflections = max_reflections

    def _step_function(self, sample: np.ndarray, random_state: np.random.Generator) -> np.ndarray:
        """Generates the next Billiard Walk sample"""
        candidate_sample = None

        while candidate_sample is None:
            trajectory_length = -self.tau * np.log(random_state.random())

            direction = random_state.standard_normal(size=sample.shape)
            direction /= np.linalg.norm(direction)

            candidate_sample = self._run_billiard_trajectory(sample, direction, trajectory_length)

        return candidate_sample

    def _run_billiard_trajectory(self,
        starting_point: np.ndarray, direction: np.ndarray, trajectory_length: float
    ) -> Optional[np.ndarray]:
        """Computes the end point of a billiard trajectory. Returns None in case
        the number of reflections attains the max_reflections limit.
        """
        num_reflections = 0
        end_point = starting_point.copy()

        while num_reflections < self.max_reflections and trajectory_length > 0:
            internal_normal, distance = self._body.compute_boundary_reflection(end_point, direction)
            distance = min(distance, trajectory_length)

            end_point += distance * direction
            direction -= 2 * internal_normal.dot(direction) * internal_normal

            num_reflections += 1
            trajectory_length -= distance

        if num_reflections == self.max_reflections:
            end_point = None

        return end_point
