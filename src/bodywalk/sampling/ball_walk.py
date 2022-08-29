from functools import partial

import numpy as np
from numpy.typing import ArrayLike

from .markov import MarkovChain

from ..geometry import ConvexBody


def ball_walk(body: ConvexBody,
              initial_point: ArrayLike,
              delta: float = 1) -> MarkovChain:
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

    step_function = partial(ball_walk_step, delta=delta)
    return MarkovChain(step_function, body, initial_point)


def ball_walk_step(body: ConvexBody,
                   sample: np.ndarray,
                   random_state: np.random.Generator,
                   delta: float) -> np.ndarray:
    """Generates the next Ball Walk sample
    """
    sample_candidate = sample_uniformly_from_ball(sample, delta, random_state)

    if body.is_inside(sample_candidate):
        sample = sample_candidate

    return sample


def sample_uniformly_from_ball(center: np.array,
                               radius: float,
                               random_state: np.random.Generator) -> np.ndarray:
    """Computes a uniformly random sample from a ball of given center and radius.
    """
    exp = 1 / center.shape[0]
    direction = random_state.standard_normal(size=center.shape)
    norm = (radius / np.linalg.norm(direction)) * random_state.random() ** exp

    return center + norm * direction
