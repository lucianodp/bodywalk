from unittest.mock import MagicMock

import pytest
import numpy as np

from bodywalk.sampling import ball_walk

from base import SamplerTestClass, SQUARE


class TestBallWalk(SamplerTestClass):
    @pytest.fixture(scope='class')
    def sampler(self):
        return ball_walk

    def test_negative_delta_throws_exception(self):
        with pytest.raises(ValueError):
            ball_walk(SQUARE, [0, 0], delta=-1)

    def test_zero_delta_throws_exception(self):
        with pytest.raises(ValueError):
            ball_walk(SQUARE, [0, 0], delta=0)

    def test_ball_walk_over_square(self):
        delta = 0.2

        random_state = MagicMock(spec=np.random.Generator)
        random_state.standard_normal.side_effect = np.array([
            [1, 0],
            [0, -1],
            [-1, 0],
            [0, 1],
        ])

        random_state.random.side_effect = [
            0.16,
            0.25,
            0.64,
            0.81,
        ]

        chain = ball_walk(SQUARE, [0, 0], random_state, delta)
        samples = chain.sample(4)

        np.testing.assert_allclose(samples, np.array([
            [ 0.08,  0.  ],
            [ 0.08, -0.1 ],
            [-0.08, -0.1 ],
            [-0.08,  0.08]
        ]))
