import pytest
import numpy as np

from bodywalk.geometry import Polytope, Ball


SQUARE = Polytope([[1, 0], [-1, 0], [0, 1], [0, -1]], [0.5, 0.5, 0.5, 0.5])
UNIT_BALL = Ball([0, 0], 1)


class SamplerTestClass:
    def test_exception_is_raised_if_initial_sample_and_convex_body_have_incompatible_dims(self, sampler):
        with pytest.raises(ValueError):
            sampler(SQUARE, [0, 0, 0])

    def test_equal_seeds_return_identical_chains(self, sampler):
        chain = sampler(SQUARE, [0, 0])

        np.testing.assert_allclose(
            chain.sample(5, random_state=1),
            chain.sample(5, random_state=1)
        )

    def test_distinct_seeds_return_distinct_chains(self, sampler):
        chain = sampler(SQUARE, [0, 0])

        assert not np.allclose(
            chain.sample(5, random_state=1),
            chain.sample(5, random_state=2)
        )

    def test_initial_sample_remains_unaltered_after_sampling(self, sampler):
        initial_sample = np.zeros(2)

        chain = sampler(SQUARE, initial_sample)
        chain.sample(1)

        np.testing.assert_allclose(initial_sample, np.zeros(2))
        np.testing.assert_allclose(chain._initial_sample, np.zeros(2))

    @pytest.mark.parametrize("body", [SQUARE, UNIT_BALL])
    def test_generated_samples_are_inside_the_convex_body(self, sampler, body):
        chain = sampler(body, [0, 0])
        samples = chain.sample(5)

        for i in range(5):
            assert body.is_inside(samples[i])
