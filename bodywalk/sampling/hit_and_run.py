from typing import Generator

import numpy as np
from numpy.typing import ArrayLike

from .utils import RandomStateLike, check_random_state
from ..geometry import ConvexBody


def hit_and_run(body: ConvexBody,
                initial_point: ArrayLike,
                random_state: RandomStateLike = None) -> Generator[np.ndarray, None, None]:
    """Hit-and-Run samplers for general convex bodies.
    """
    random_state = check_random_state(random_state)

    sample = np.array(initial_point, dtype='float', copy=True)

    while True:
        random_direction = random_state.normal(size=sample.shape)

        lower, upper = body.compute_intersection_extremes(sample, random_direction)
        if lower >= upper:
            raise ValueError(f"Lower extreme must be smaller than upper extreme, but {lower} >= {upper}")

        sample += random_state.uniform(lower, upper) * random_direction

        yield sample.copy()  # return a copy because 'sample' is modified at every iteration
