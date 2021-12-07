from typing import Generator

import numpy as np
from numpy.typing import ArrayLike

from .utils import RandomStateLike, check_random_state
from ..geometry import ConvexBody


def ball_walk(body: ConvexBody,
              initial_point: ArrayLike,
              random_state: RandomStateLike = None,
              delta: float = 1) -> Generator[np.ndarray, None, None]:
    """Ball walk samplers for general convex bodies.
    """
    if delta <= 0:
        raise ValueError(f"Expected a positive value for 'delta', but got {delta}")

    random_state = check_random_state(random_state)

    sample = np.array(initial_point, dtype='float', copy=True)

    while True:
        sample_candidate = sample_uniformly_from_ball(sample, delta, random_state)

        if body.is_inside(sample_candidate):
            sample = sample_candidate

        yield sample


def sample_uniformly_from_ball(center: np.array, radius: float, random_state: np.random.RandomState):
    """Computes a uniformly random sample from a ball of given center and radius.
    """
    exp = 1 / center.shape[0]
    direction = random_state.normal(size=center.shape)
    norm = (radius / np.linalg.norm(direction)) * random_state.rand() ** exp

    return center + norm * direction
