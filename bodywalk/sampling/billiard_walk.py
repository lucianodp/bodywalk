from typing import Generator, Optional

import numpy as np
from numpy.typing import ArrayLike

from .utils import RandomStateLike, check_random_state
from ..geometry import ConvexBody


def billiard_walk(body: ConvexBody,
                  initial_point: ArrayLike,
                  random_state: RandomStateLike = None,
                  tau: float = 1,
                  max_reflections: Optional[int] = None) -> Generator[np.ndarray, None, None]:
    """Generate a Markov Chain inside a convex body converging to the uniform distribution
    via the Billiard Walk algorithm.

    Parameters
    ----------
    body : ConvexBody
        The convex body to sample from. It requires an implementation of the
        'compute_boundary_reflection' method.
    initial_point : ArrayLike
        The starting point for the Markov chain. It must be inside the convex body.
    random_state : None (default), int, or np.random.RandomState instance
        The random number generator instance. It can be specified in 3 ways:
            - None: creates a new RandomState instance with unspecified seed
            - int: seed to be used for the RNG. Allows for reproducibility.
            - RandomState instance: a specific RandomState instance to be used
    tau : float (default = 1)
        Average length of billiard trajectory.
    max_reflections: Optional[int] (default = None)
        Maximum number of reflections allowed per billiard trajectory. If None,
        an heuristic of 10 * n will be used, n being the dimensionality.


    Yields
    -------
    Generator[np.ndarray, None, None]
        The Markov Chain samples generated by the sampling algorithm.

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
    random_state = check_random_state(random_state)

    sample = np.array(initial_point, dtype='float', copy=True)

    if tau <= 0:
        raise ValueError(f"tau must be a positive number, but got {tau}")

    if max_reflections is None:
        max_reflections = 10 * sample.shape[0]

    if max_reflections <= 0:
        raise ValueError(f"max_reflections must be a positive integer, but got {max_reflections}")

    while True:
        num_reflections = 0
        trajectory_length = -tau * np.log(random_state.rand())

        direction = random_state.normal(size=sample.shape)
        direction /= np.linalg.norm(direction)

        candidate_sample = sample.copy()

        while num_reflections < max_reflections and trajectory_length > 0:
            internal_normal, distance = body.compute_boundary_reflection(candidate_sample, direction)
            distance = min(distance, trajectory_length)

            candidate_sample += distance * direction
            direction -= 2 * internal_normal.dot(direction) * internal_normal

            num_reflections += 1
            trajectory_length -= distance

        if num_reflections < max_reflections:
            sample = candidate_sample
            yield sample
