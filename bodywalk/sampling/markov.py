from typing import Callable, Generator

import numpy as np
from numpy.typing import ArrayLike

from .utils import RandomStateLike, check_random_state
from ..geometry import ConvexBody


StepFunction = Callable[[ConvexBody, np.ndarray, np.random.RandomState], np.ndarray]
MarkovChain = Generator[np.ndarray, None, None]


def generate_markov_chain(step_function: StepFunction,
                          body: ConvexBody,
                          initial_point: ArrayLike,
                          random_state: RandomStateLike = None) -> MarkovChain:
    """Generates a Markov Chain over a ConvexBody with the given configuration.
    """
    random_state = check_random_state(random_state)
    sample = np.array(initial_point, dtype='float', copy=True)

    while True:
        sample = step_function(body, sample, random_state)
        yield sample
