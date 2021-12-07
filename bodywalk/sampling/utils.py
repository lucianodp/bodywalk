from typing import Union

import numpy as np


RandomStateLike = Union[None, int, np.random.RandomState]


def check_random_state(random_state: RandomStateLike) -> np.random.RandomState:
    if isinstance(random_state, np.random.RandomState):
        return random_state
    return np.random.RandomState(random_state)
