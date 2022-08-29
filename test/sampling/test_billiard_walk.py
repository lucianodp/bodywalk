from unittest.mock import MagicMock
import pytest

import numpy as np

from bodywalk.sampling import billiard_walk

from base import SamplerTestClass, SQUARE


class TestBilliardWalk(SamplerTestClass):
    @pytest.fixture(scope='class')
    def sampler(self):
        return billiard_walk

    def test_negative_tau_throws_exception(self):
        with pytest.raises(ValueError):
            billiard_walk(SQUARE, [0, 0], tau=-1)

    def test_zero_tau_throws_exception(self):
        with pytest.raises(ValueError):
            billiard_walk(SQUARE, [0, 0], tau=0)

    def test_negative_max_reflections_throws_exception(self):
        with pytest.raises(ValueError):
            billiard_walk(SQUARE, [0, 0], max_reflections=-1)

    def test_zero_max_reflections_throws_exception(self):
        with pytest.raises(ValueError):
            billiard_walk(SQUARE, [0, 0], max_reflections=0)

    def test_billiard_walk_over_square(self):
        tau = 0.5

        random_state = MagicMock(spec=np.random.Generator)
        random_state.standard_normal.side_effect = np.array([
            [4, 3],
            [-3, -4],
            [-4, 3],
            [3, -4],
        ], dtype='float')

        random_state.random.side_effect = np.array([
            1 / np.e ** 4,
            1 / np.e ** 3,
            1 / np.e ** 2,
            1 / np.e,
        ])

        chain = billiard_walk(SQUARE, [0, 0], random_state, tau)
        samples = chain.sample(4)

        np.testing.assert_allclose(samples, np.array([
            [-0.4, -0.2],  # len = 2.0, d = [0.8, 0.6]
            [0.3, 0.4],    # len = 1.5, d = [-0.6, -0.8]
            [-0.5, 0],     # len = 1.0, d = [-0.8, 0.6]
            [-0.2, -0.4],  # len = 0.5, d = [0.6, -0.8]
        ], dtype='float'), atol=1e-15)
