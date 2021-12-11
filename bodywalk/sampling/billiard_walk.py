from typing import Generator, Optional

import numpy as np
from numpy.typing import ArrayLike

from .utils import RandomStateLike, check_random_state
from ..geometry import ConvexBody, Polytope


def billiard_walk(body: ConvexBody,
                  initial_point: ArrayLike,
                  random_state: RandomStateLike = None,
                  tau: float = 1,
                  max_reflections: Optional[int] = None) -> Generator[np.ndarray, None, None]:
    """Billiard walk samplers for general convex bodies.
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
