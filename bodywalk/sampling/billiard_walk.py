from functools import partial
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from .markov import MarkovChain, generate_markov_chain
from .utils import RandomStateLike

from ..geometry import ConvexBody


def billiard_walk(body: ConvexBody,
                  initial_point: ArrayLike,
                  random_state: RandomStateLike = None,
                  tau: float = 1,
                  max_reflections: Optional[int] = None) -> MarkovChain:
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
    if tau <= 0:
        raise ValueError(f"tau must be a positive number, but got {tau}")

    if max_reflections is None:
        max_reflections = 10 * body.dim

    if max_reflections <= 0:
        raise ValueError(f"max_reflections must be a positive integer, but got {max_reflections}")

    step_function = partial(billiard_walk_step_function, tau=tau, max_reflections=max_reflections)
    return generate_markov_chain(step_function, body, initial_point, random_state)


def billiard_walk_step_function(body: ConvexBody,
                    sample: np.ndarray,
                    random_state: np.random.RandomState,
                    tau: float,
                    max_reflections: int) -> np.ndarray:
    """Performs the billiard walk algorithm step
    """
    candidate_sample = None

    while candidate_sample is None:
        trajectory_length = -tau * np.log(random_state.rand())

        direction = random_state.normal(size=sample.shape)
        direction /= np.linalg.norm(direction)

        candidate_sample = run_billiard_trajectory(body, sample, direction, trajectory_length, max_reflections)

    return candidate_sample


def run_billiard_trajectory(body: ConvexBody,
                            starting_point: np.ndarray,
                            direction: np.ndarray,
                            trajectory_length: float,
                            max_reflections: int) -> Optional[np.ndarray]:
    """Computes the end point of a billiard trajectory. Returns None in case
    the number of reflections attains the max_reflections limit.
    """
    num_reflections = 0
    end_point = starting_point.copy()

    while num_reflections < max_reflections and trajectory_length > 0:
        internal_normal, distance = body.compute_boundary_reflection(end_point, direction)
        distance = min(distance, trajectory_length)

        end_point += distance * direction
        direction -= 2 * internal_normal.dot(direction) * internal_normal

        num_reflections += 1
        trajectory_length -= distance

    if num_reflections == max_reflections:
        end_point = None

    return end_point
