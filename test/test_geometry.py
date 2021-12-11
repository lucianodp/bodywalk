from abc import abstractmethod
import pytest

import numpy as np

from bodywalk.geometry import Polytope, Ball


TRIANGLE = Polytope([[-1, 0], [0, -1], [1, 1]], [0, 0, 1])
DIAMOND = Polytope([[1, 1], [1, -1], [-1, -1], [-1, 1]], [1, 1, 1, 1])
UNIT_BALL = Ball(np.zeros(2), 1)
GENERAL_BALL = Ball([1, 2], 3)


class ConvexBodyTestClass:
    def test_compute_intersection(self, body, x, v, expected_lower, expected_upper):
        lower, upper = body.compute_intersection_extremes(np.array(x), np.array(v))

        assert lower == pytest.approx(expected_lower)
        assert upper == pytest.approx(expected_upper)

    def test_boundary_reflection(self, body, x, v, expected_normal, expected_dist):
        x, v = np.array(x), np.array(v)

        internal_normal, distance = body.compute_boundary_reflection(x, v)

        np.testing.assert_allclose(np.array(internal_normal), expected_normal)
        assert distance == pytest.approx(expected_dist)



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

    @pytest.mark.parametrize("pol, x, is_inside", [
        (TRIANGLE, [0.2, 0.5], True),  # interior
        (TRIANGLE, [0.4, 0.5], True),  # interior
        (TRIANGLE, [0.8, 0.1], True),  # interior
        (TRIANGLE, [0, 0], True),  # edge
        (TRIANGLE, [1, 0], True),  # edge
        (TRIANGLE, [0, 1], True),  # edge
        (TRIANGLE, [-0.1, 0.5], False),  # outside
        (TRIANGLE, [0.1, -0.5], False),  # outside
        (TRIANGLE, [0.6, 0.6], False),  # outside
    ])
    def test_is_inside(self, pol, x, is_inside):
        assert pol.is_inside(x) == is_inside

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

    @pytest.mark.parametrize("body, x, v, expected_normal, expected_dist", [
        (TRIANGLE, [0.2, 0.4], [-1, 0], [1, 0], 0.2),
        (TRIANGLE, [0.2, 0.4], [0, -1], [0, 1], 0.4),
        (TRIANGLE, [0.2, 0.4], [0.6, 0.8], [-1/np.sqrt(2), -1/np.sqrt(2)], 2/7),
    ])
    def test_boundary_reflection(self, body, x, v, expected_normal, expected_dist):
        return super().test_boundary_reflection(body, x, v, expected_normal, expected_dist)


class TestBall(ConvexBodyTestClass):
    def test_negative_radius_throws_exception(self):
        with pytest.raises(ValueError):
            Ball(np.zeros(2), -2)

    def test_zero_radius_throws_exception(self):
        with pytest.raises(ValueError):
            Ball(np.zeros(2), 0)

    @pytest.mark.parametrize("ball, x, is_inside", [
        (UNIT_BALL, [0.2, 0.5], True), # interior
        (UNIT_BALL, [1, 0], True),     # boundary
        (UNIT_BALL, [0, -2], False),   # exterior
        (GENERAL_BALL, [3, 4], True),   # interior
        (GENERAL_BALL, [4, 2], True),   # boundary
        (GENERAL_BALL, [5, -6], False), # exterior
    ])
    def test_is_inside(self, ball, x, is_inside):
        assert ball.is_inside(x) == is_inside

    @pytest.mark.parametrize("ball, x, v, expected_lower, expected_upper", [
        (UNIT_BALL, [0, 0], [1, 0], -1, 1),
        (UNIT_BALL, [0, 0], [0, 2], -0.5, 0.5),
        (UNIT_BALL, [0, 0], [-3, 4], -0.2, 0.2),
        (GENERAL_BALL, [1, 2], [-3, -4], -0.6, 0.6),
        (GENERAL_BALL, [2, 3], [-1, 2], -1.4, 1),
    ])
    def test_compute_intersection(self, ball, x, v, expected_lower, expected_upper):
        super().test_compute_intersection(ball, x, v, expected_lower, expected_upper)

    @pytest.mark.parametrize("body, x, v, expected_normal, expected_dist", [
        (UNIT_BALL, [0, 0], [1, 0], [-1, 0], 1),
        (UNIT_BALL, [0.3, -0.4], [0.6, -0.8], [-0.6, 0.8], 0.5),
        (UNIT_BALL, [0.6, -0.3], [0.8, 0.6], [-1, 0], 0.5),
    ])
    def test_boundary_reflection(self, body, x, v, expected_normal, expected_dist):
        return super().test_boundary_reflection(body, x, v, expected_normal, expected_dist)
