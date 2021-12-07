from typing import Generator, Union

import numpy as np
from numpy.typing import ArrayLike

from ..geometry import ConvexBody


RandomStateLike = Union[None, int, np.random.RandomState]


def check_random_state(random_state: RandomStateLike) -> np.random.RandomState:
    if isinstance(random_state, np.random.RandomState):
        return random_state
    return np.random.RandomState(random_state)


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

        sample = sample + random_state.uniform(lower, upper) * random_direction

        yield sample
