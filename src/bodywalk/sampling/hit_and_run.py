from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from .markov import MarkovChain
from ..geometry import ConvexBody


class hit_and_run(MarkovChain):

    def __init__(
        self,
        body: ConvexBody, initial_point: ArrayLike,
        rounding_matrix: Optional[ArrayLike] = None
    ):
        """Generate a Markov Chain inside a convex body converging to the uniform distribution
        via the Hit-and-Run Walk algorithm.

        Parameters
        ----------
        body : ConvexBody
            The convex body to sample from. It requires an implementation of the
            compute_intersection_extremes method.
        initial_point : ArrayLike
            The starting point for the Markov chain. It must be inside the convex body.
        rounding_matrix : None (default) or ArrayLike
            Matrix to be applied to random direction samples. For "very elongated" convex bodies,
            passing a sensible rounding matrix may drastically reduce the mixing time. By default, no
            rounding will be applied. See Rounding Algorithm for more information.

        Returns
        -------
        MarkovChain
            A MarkovChain object that generates samples form the convex body via the Hit-and-Run strategy.

        Raises
        ------
        ValueError
            If invalid line segment is computed (lower extreme >= upper extreme).

        References
        ----------
            [1] C. J. P. Bélisle, H. E. Romeijn, R. L. Smith. Hit-and-Run Algorithms for Generating Multivariate Distributions
                Mathematics of Operations Research, Vol. 18, No. 2. 1993

            [2] L. Lovasz and S. Vempala. Hit-and-Run is Fast and Fun.
                Technical Report. 2003.
                https://web.cs.elte.hu/~lovasz/logcon-hitrun.pdf
        """
        if rounding_matrix is not None:
            rounding_matrix = np.asarray(rounding_matrix)

            if rounding_matrix.ndim != 2:
                raise ValueError(
                    f"rounding_matrix must be a 2-dimensional array, but ndim={rounding_matrix.ndim}"
                )

            if rounding_matrix.shape != (body.dim, body.dim):
                raise ValueError(
                    f'expected rounding_matrix to be of shape {(body.dim, body.dim)},'
                    ' but got {rounding_matrix.shape}'
                )

        super().__init__(body, initial_point)
        self.rounding_matrix = rounding_matrix

    def _step_function(self, sample: np.ndarray, random_state: np.random.Generator) -> np.ndarray:
        """Generates the next hit-and-run sample
        """
        random_direction = random_state.standard_normal(size=sample.shape)

        if self.rounding_matrix is not None:
            random_direction = self.rounding_matrix.dot(random_direction)

        lower, upper = self._body.compute_intersection_extremes(sample, random_direction)
        if lower >= upper:
            raise RuntimeError(
                f"Lower extreme must be smaller than upper extreme, but got {lower} >= {upper}"
            )

        return sample + random_state.uniform(lower, upper) * random_direction
