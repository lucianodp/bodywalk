from unittest.mock import MagicMock
import pytest

import numpy as np

from bodywalk.geometry import ConvexBody
from bodywalk.sampling import hit_and_run

from base import SamplerTestClass, SQUARE


class TestHitAndRun(SamplerTestClass):
    @pytest.fixture(scope='class')
    def sampler(self):
        return hit_and_run

    def test_exception_is_raised_if_lower_extreme_larger_than_upper(self):
        body = MagicMock(spec=ConvexBody)
        body.dim = 2
        body.compute_intersection_extremes.return_value = (1, -1)

        chain = hit_and_run(body, [0, 0])

        with pytest.raises(RuntimeError):
            next(chain)

    def test_exception_is_raised_if_lower_extreme_equals_upper(self):
        body = MagicMock(spec=ConvexBody)
        body.dim = 2
        body.compute_intersection_extremes.return_value = (1, 1)

        chain = hit_and_run(body, [0, 0])

        with pytest.raises(RuntimeError):
            next(chain)

    def test_hit_and_run_over_square(self):
        random_state = MagicMock(spec=np.random.Generator)
        random_state.standard_normal.side_effect = np.array([
            [1, 0],
            [0, -1],
            [-1, 0],
            [0, 1],
        ])

        random_state.uniform.side_effect = np.array([
            +0.2,  # [-0.5, 0.5]
            -0.4,  # [-0.5, 0.5]
            +0.3,  # [-0.4, 0.6]
            -0.5,  # [-0.7, 0.3]
        ])

        chain = hit_and_run(SQUARE, [0, 0], random_state)
        samples = [next(chain) for _ in range(4)]

        np.testing.assert_allclose(samples, np.array([
            [0.2, 0.0],
            [0.2, 0.4],
            [-0.1, 0.4],
            [-0.1, -0.1],
        ]))
