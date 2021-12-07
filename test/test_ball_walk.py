from unittest.mock import MagicMock
import pytest

import numpy as np

from bodywalk.geometry import ConvexBody, Polytope
from bodywalk.sampling import ball_walk


SQUARE = Polytope([[1, 0], [-1, 0], [0, 1], [0, -1]], [1, 1, 1, 1])


class TestBallWalk:
    def test_equal_seeds_return_identical_chains(self):
        seed = 1
        chain1 = ball_walk(SQUARE, np.zeros(2), seed)
        chain2 = ball_walk(SQUARE, np.zeros(2), seed)

        for _ in range(5):
            np.testing.assert_allclose(next(chain1), next(chain2))

    def test_None_seeds_return_distinct_chains(self):
        seed = None
        chain1 = ball_walk(SQUARE, np.zeros(2), seed)
        chain2 = ball_walk(SQUARE, np.zeros(2), seed)

        for _ in range(5):
            assert (next(chain1) != next(chain2)).any()


    def test_negative_delta_throws_exception(self):
        with pytest.raises(ValueError):
            chain = ball_walk(SQUARE, np.zeros(2), delta=-1)
            next(chain)

    def test_zero_delta_throws_exception(self):
        with pytest.raises(ValueError):
            chain = ball_walk(SQUARE, np.zeros(2), delta=0)
            next(chain)

    def test_ball_walk_over_square(self):
        delta = 0.5

        random_state = MagicMock(spec=np.random.RandomState)
        random_state.normal.side_effect = np.array([
            [1, 0],
            [0, -1],
            [-1, 0],
            [0, 1],
        ])

        random_state.rand.side_effect = np.array([
            0.16,
            0.25,
            0.64,
            0.81,
        ])

        chain = ball_walk(SQUARE, np.zeros(2), random_state, delta)
        samples = [next(chain) for _ in range(4)]

        np.testing.assert_allclose(samples, np.array([
            [0.2, 0.0],
            [0.2, -0.25],
            [-0.2, -0.25],
            [-0.2, 0.2],
        ]))
