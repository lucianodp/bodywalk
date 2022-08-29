import numpy as np
from numpy.typing import ArrayLike

from .markov import MarkovChain

from ..geometry import ConvexBody

class ball_walk(MarkovChain):

    def __init__(
        self,
        body: ConvexBody,
        initial_point: ArrayLike,
        delta: float = 1
    ) -> MarkovChain:
        """Generate a Markov Chain inside a convex body converging to the uniform distribution
        via the Ball Walk algorithm.

        Parameters
        ----------
        body : ConvexBody
            The convex body to sample from. It requires an implementation of the is_inside method.
        initial_point : ArrayLike
            The starting point for the Markov chain. It must be inside the convex body.
        delta : float (default = 1)
            Radius of the ball samples at each iteration of the algorithm.

        Returns
        -------
        MarkovChain
            A MarkovChain object that generates samples form the convex body via the Ball Walk strategy.

        Raises
        ------
        ValueError
            If delta is negative or zero.

        References
        ----------
            [1] S. Vempala. Geometric Random Walks: A Survey.
                Combinatorial and Computational Geometry, Volume 52, 2005
                https://www.cc.gatech.edu/~vempala/papers/survey.pdf

            [2] L. Lovasz and M. Simonovits. Random walks in a convex body and an improved volume algorithm.
                Random Structures and Algorithms, 4:359–412, 1993.
                http://matmod.elte.hu/~lovasz/vol7.pdf
        """
        if delta <= 0:
            raise ValueError(f"Expected a positive value for 'delta', but got {delta}")

        super().__init__(body, initial_point)
        self.delta = delta


    def _step_function(self, sample: np.ndarray, random_state: np.random.Generator) -> np.ndarray:
        """Generates the next Ball Walk sample"""
        sample_candidate = self._sample_uniformly_from_ball(sample, random_state)

        if self._body.is_inside(sample_candidate):
            sample = sample_candidate

        return sample


    def _sample_uniformly_from_ball(self, center: np.array, random_state: np.random.Generator) -> np.ndarray:
        """Computes a uniformly random sample from a ball of given center and radius."""
        exp = 1 / center.shape[0]
        direction = random_state.standard_normal(size=center.shape)
        norm = (self.delta / np.linalg.norm(direction)) * random_state.random() ** exp

        return center + norm * direction
