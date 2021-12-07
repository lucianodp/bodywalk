from unittest.mock import MagicMock
import pytest

import numpy as np

from bodywalk.geometry import ConvexBody, Polytope
from bodywalk.sampling import hit_and_run


SQUARE = Polytope([[1, 0], [-1, 0], [0, 1], [0, -1]], [1, 1, 1, 1])


class TestHitAndRun:
    def test_exception_is_raised_if_lower_extreme_larger_than_upper(self):
        body = MagicMock(spec=ConvexBody)
        body.compute_intersection_extremes.return_value = (1, -1)

        chain = hit_and_run(body, np.zeros(2))

        with pytest.raises(ValueError):
            next(chain)

    def test_exception_is_raised_if_lower_extreme_equals_upper(self):
        body = MagicMock(spec=ConvexBody)
        body.compute_intersection_extremes.return_value = (1, 1)

        chain = hit_and_run(body, np.zeros(2))

        with pytest.raises(ValueError):
            next(chain)

    def test_equal_seeds_return_identical_chains(self):
        seed = 1
        chain1 = hit_and_run(SQUARE, np.zeros(2), seed)
        chain2 = hit_and_run(SQUARE, np.zeros(2), seed)

        for _ in range(5):
            np.testing.assert_allclose(next(chain1), next(chain2))

    def test_None_seeds_return_distinct_chains(self):
        seed = None
        chain1 = hit_and_run(SQUARE, np.zeros(2), seed)
        chain2 = hit_and_run(SQUARE, np.zeros(2), seed)

        for _ in range(5):
            assert (next(chain1) != next(chain2)).any()


    def test_hit_and_run_over_square(self):
        random_state = MagicMock(spec=np.random.RandomState)
        random_state.normal.side_effect = np.array([
            [1, 0],
            [0, -1],
            [-1, 0],
            [0, 1],
        ])

        random_state.uniform.side_effect = np.array([
            +0.2,  # [-1, 1]
            -0.4,  # [-1, 1]
            +0.9,  # [-0.8, 1.2]
            -1.2,  # [-1.4, 0.6]
        ])

        chain = hit_and_run(SQUARE, np.zeros(2), random_state)
        samples = [next(chain) for _ in range(4)]

        np.testing.assert_allclose(samples, np.array([
            [0.2, 0.0],
            [0.2, 0.4],
            [-0.7, 0.4],
            [-0.7, -0.8],
        ]))
