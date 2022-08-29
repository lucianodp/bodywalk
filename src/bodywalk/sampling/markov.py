from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from .utils import RandomStateLike, check_random_state
from ..geometry import ConvexBody


StepFunction = Callable[[ConvexBody, np.ndarray, np.random.Generator], np.ndarray]

class MarkovChain:
    """Class representing a Markov Chain over a ConvexBody."""

    def __init__(self, step_function: StepFunction,
                 body: ConvexBody,
                 initial_point: ArrayLike,
                 random_state: RandomStateLike = None):
        """
        Parameters
        ----------
        step_function : StepFunction
            Function generating the next sample of the Markov Chain
        body : ConvexBody
            ConvexBody object to sample from
        initial_point : ArrayLike
            Initial point for Markov Chain
        random_state : RandomStateLike, optional
            Random state generator object, by default None

        Raises
        ------
        ValueError
            If initial point and body dimensionality are incompatible
        """
        initial_point = np.array(initial_point, dtype='float', copy=True)

        if initial_point.size != body.dim:
            raise ValueError(
                f"Convex body and initial sample have incompatible sizes: \
                {body.dim} != {initial_point.size}"
            )

        self._body = body
        self._initial_sample = initial_point
        self._step_function = step_function
        self._random_state = check_random_state(random_state)

    @property
    def dim(self) -> int:
        """
        Returns
        -------
        int
            Underlying dimensionality of convex body / samples.
        """
        return self._body.dim

    def generate(self):
        current_sample = self._initial_sample
        while True:
            current_sample = self.__advance(current_sample, n=1)
            yield current_sample

    def sample(self, n: int = 1, warmup: int = 1, thin: int = 1) -> np.ndarray:
        if n <= 0:
            raise ValueError(f"Number of samples 'n' must be positive, but got {n}")

        if warmup < 0:
            raise ValueError(f"'warmup' value must be positive, but got {warmup}")

        if thin <= 0:
            raise ValueError(f"'thin' value must be positive, but got {thin}")

        samples = np.empty((n, self.dim))

        samples[0] = self.__advance(self._initial_sample, warmup)
        for i in range(1, n):
            samples[i] = self.__advance(samples[i-1], thin)

        return samples

    def __advance(self, current_sample: np.ndarray, n: int) -> None:
        """Advances the Markov Chain to the next sample."""
        for _ in range(n):
            current_sample = self._step_function(
                self._body, current_sample, self._random_state
            )

        return current_sample
