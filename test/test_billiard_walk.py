from unittest.mock import MagicMock
import pytest

import numpy as np

from bodywalk.geometry import Polytope
from bodywalk.sampling import billiard_walk


SQUARE = Polytope([[1, 0], [-1, 0], [0, 1], [0, -1]], [0.5, 0.5, 0.5, 0.5])


class TestBilliardWalk:
    def test_equal_seeds_return_identical_chains(self):
        seed = 1
        chain1 = billiard_walk(SQUARE, np.zeros(2), seed)
        chain2 = billiard_walk(SQUARE, np.zeros(2), seed)

        for _ in range(5):
            np.testing.assert_allclose(next(chain1), next(chain2))

    def test_None_seeds_return_distinct_chains(self):
        seed = None
        chain1 = billiard_walk(SQUARE, np.zeros(2), seed)
        chain2 = billiard_walk(SQUARE, np.zeros(2), seed)

        for _ in range(5):
            assert (next(chain1) != next(chain2)).any()

    def test_negative_tau_throws_exception(self):
        with pytest.raises(ValueError):
            chain = billiard_walk(SQUARE, np.zeros(2), tau=-1)
            next(chain)

    def test_zero_tau_throws_exception(self):
        with pytest.raises(ValueError):
            chain = billiard_walk(SQUARE, np.zeros(2), tau=0)
            next(chain)

    def test_negative_max_reflections_throws_exception(self):
        with pytest.raises(ValueError):
            chain = billiard_walk(SQUARE, np.zeros(2), max_reflections=-1)
            next(chain)

    def test_zero_max_reflections_throws_exception(self):
        with pytest.raises(ValueError):
            chain = billiard_walk(SQUARE, np.zeros(2), max_reflections=0)
            next(chain)

    def test_billiard_walk_over_square(self):
        tau = 0.5

        random_state = MagicMock(spec=np.random.RandomState)
        random_state.normal.side_effect = np.array([
            [4, 3],
            [-3, -4],
            [-4, 3],
            [3, -4],
        ], dtype='float')

        random_state.rand.side_effect = np.array([
            1 / np.e ** 4,
            1 / np.e ** 3,
            1 / np.e ** 2,
            1 / np.e,
        ])

        chain = billiard_walk(SQUARE, np.zeros(2), random_state, tau)
        samples = [next(chain) for _ in range(4)]

        np.testing.assert_allclose(samples, np.array([
            [-0.4, -0.2],  # len = 2.0, d = [0.8, 0.6]
            [0.3, 0.4],    # len = 1.5, d = [-0.6, -0.8]
            [-0.5, 0],     # len = 1.0, d = [-0.8, 0.6]
            [-0.2, -0.4],  # len = 0.5, d = [0.6, -0.8]
        ], dtype='float'), atol=1e-15)
