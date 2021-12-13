from typing import Union

import numpy as np


RandomStateLike = Union[None, int, np.random.Generator]


def check_random_state(random_state: RandomStateLike) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)
