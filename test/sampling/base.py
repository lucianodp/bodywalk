import pytest
import numpy as np

from bodywalk.geometry import Polytope, Ball


SQUARE = Polytope([[1, 0], [-1, 0], [0, 1], [0, -1]], [0.5, 0.5, 0.5, 0.5])
UNIT_BALL = Ball([0, 0], 1)


class SamplerTestClass:
    def test_exception_is_raised_if_initial_sample_and_convex_body_have_incompatible_dims(self, sampler):
        with pytest.raises(ValueError):
            chain = sampler(SQUARE, [0, 0, 0])
            next(chain)

    def test_equal_seeds_return_identical_chains(self, sampler):
        chain1 = sampler(SQUARE, [0, 0], random_state=1)
        chain2 = sampler(SQUARE, [0, 0], random_state=1)

        for _ in range(5):
            np.testing.assert_allclose(next(chain1), next(chain2))

    def test_distinct_seeds_return_distinct_chains(self, sampler):
        chain1 = sampler(SQUARE, [0, 0], random_state=1)
        samples1 = np.array([next(chain1) for _ in range(5)])

        chain2 = sampler(SQUARE, [0, 0], random_state=2)
        samples2 = np.array([next(chain2) for _ in range(5)])

        assert not np.allclose(samples1, samples2)

    @pytest.mark.parametrize("body", [SQUARE, UNIT_BALL])
    def test_generated_samples_are_inside_the_convex_body(self, sampler, body):
        chain = sampler(body, [0, 0])

        for _ in range(5):
            assert body.is_inside(next(chain))
