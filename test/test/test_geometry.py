from abc import abstractmethod
import pytest

import numpy as np

from bodywalk.geometry import Polytope, Ball


TRIANGLE = Polytope([[-1, 0], [0, -1], [1, 1]], [0, 0, 1])
DIAMOND = Polytope([[1, 1], [1, -1], [-1, -1], [-1, 1]], [1, 1, 1, 1])


class ConvexBodyTestClass:
    def test_compute_intersection(self, body, x, v, expected_lower, expected_upper):
        lower, upper = body.compute_intersection_extremes(np.array(x), np.array(v))

        assert lower == pytest.approx(expected_lower)
        assert upper == pytest.approx(expected_upper)


class TestPolytope(ConvexBodyTestClass):
    def test_one_dimensional_A_throws_exception(self):
        with pytest.raises(ValueError):
            Polytope(A=[1, 2], b=[3, 4])

    def test_three_dimensional_A_throws_exception(self):
        with pytest.raises(ValueError):
            Polytope(A=[[[1, 2]]], b=[3, 4])

    def test_two_dimensional_b_throws_exception(self):
        with pytest.raises(ValueError):
            Polytope(A=[[1, 2]], b=[[3, 4]])

    def test_incompatible_dimensions_for_A_and_b_throws_exception(self):
        with pytest.raises(ValueError):
            Polytope(A=[[1, 2], [3, 4]], b=[5])

    @pytest.mark.parametrize("pol, x, v, expected_lower, expected_upper", [
        (TRIANGLE, [0.2, 0.7], [1, 0], -0.2, 0.1),
        (TRIANGLE, [0.8, 0.1], [0, -2], -0.05, 0.05),
        (TRIANGLE, [0.3, 0.3], [1, 3], -0.1, 0.1),
        (DIAMOND, [0, 0], [1, 0], -1, 1),
        (DIAMOND, [0, -0.5], [0, -2], -0.75, 0.25),
        (DIAMOND, [0.2, 0.4], [1, 3], -0.4, 0.1),
    ])
    def test_compute_intersection(self, pol, x, v, expected_lower, expected_upper):
        super().test_compute_intersection(pol, x, v, expected_lower, expected_upper)


class TestBall(ConvexBodyTestClass):
    def test_negative_radius_throws_exception(self):
        with pytest.raises(ValueError):
            Ball(np.zeros(2), -2)

    def test_zero_radius_throws_exception(self):
        with pytest.raises(ValueError):
            Ball(np.zeros(2), 0)

    @pytest.mark.parametrize("ball, x, v, expected_lower, expected_upper", [
        (Ball([0, 0], 1), [0, 0], [1, 0], -1, 1),
        (Ball([0, 0], 1), [0, 0], [0, 2], -0.5, 0.5),
        (Ball([0, 0], 1), [0, 0], [-3, 4], -0.2, 0.2),
        (Ball([0, 0], 3), [0, 0], [3, -4], -0.6, 0.6),
        (Ball([-2, -3], 2), [-2, -3], [-3, -4], -0.4, 0.4),
        (Ball([1, 2], 3), [2, 3], [-1, 2], -1.4, 1),
    ])
    def test_compute_intersection(self, ball, x, v, expected_lower, expected_upper):
        super().test_compute_intersection(ball, x, v, expected_lower, expected_upper)
