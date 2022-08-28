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

    def test_exception_is_raised_if_rounding_matrix_is_not_two_dimensional(self):
        body = MagicMock(spec=ConvexBody)

        with pytest.raises(ValueError):
            hit_and_run(body, [0, 0], rounding_matrix=[1])

        with pytest.raises(ValueError):
            hit_and_run(body, [0, 0], rounding_matrix=[[[1]]])

    def test_exception_is_raised_if_rounding_matrix_is_not_square(self):
        body = MagicMock(spec=ConvexBody)

        with pytest.raises(ValueError):
            hit_and_run(body, [0, 0], rounding_matrix=np.eye(2, 3))

    def test_exception_is_raised_if_rounding_matrix_shape_does_not_match_convex_body_dim(self):
        body = MagicMock(spec=ConvexBody)
        body.dim = 2

        with pytest.raises(ValueError):
            hit_and_run(body, [0, 0], rounding_matrix=np.eye(3))

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

        chain = hit_and_run(SQUARE, [0, 0], random_state=random_state)
        samples = [next(chain) for _ in range(4)]

        np.testing.assert_allclose(samples, np.array([
            [0.2, 0.0],
            [0.2, 0.4],
            [-0.1, 0.4],
            [-0.1, -0.1],
        ]))

    def test_hit_and_run_with_identity_rounding_matrix_has_no_effect(self):
        chain = hit_and_run(SQUARE, [0, 0], random_state=42)
        samples_without_rounding = [next(chain) for _ in range(4)]

        chain = hit_and_run(SQUARE, [0, 0], rounding_matrix=np.eye(2), random_state=42)
        samples_with_rounding = [next(chain) for _ in range(4)]

        assert np.allclose(samples_with_rounding, samples_without_rounding)

    def test_hit_and_run_with_rounding_over_square(self):
        rounding_matrix = [[2, 0], [0, 0.5]]

        random_state = MagicMock(spec=np.random.Generator)
        random_state.standard_normal.side_effect = np.array([
            [1, 0],
            [0, -1],
            [-1, 0],
            [0, 1],
        ])

        random_state.uniform.side_effect = np.array([
            +0.1,  # [-0.25, 0.25]
            -0.8,  # [-1.0, 1.0]
            +0.15, # [-0.2, 0.3]
            -1.0,  # [-1.4, 0.6]
        ])

        chain = hit_and_run(SQUARE, [0, 0], rounding_matrix=rounding_matrix, random_state=random_state)
        samples = [next(chain) for _ in range(4)]

        np.testing.assert_allclose(samples, np.array([
            [0.2, 0.0],
            [0.2, 0.4],
            [-0.1, 0.4],
            [-0.1, -0.1],
        ]))
