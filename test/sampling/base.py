import pytest
import numpy as np

from bodywalk.geometry import Polytope, Ball


SQUARE = Polytope([[1, 0], [-1, 0], [0, 1], [0, -1]], [0.5, 0.5, 0.5, 0.5])
UNIT_BALL = Ball([0, 0], 1)


class SamplerTestClass:
    @pytest.fixture(scope='class')
    def chain(self, sampler):
        return sampler(SQUARE, [0, 0])

    def test_exception_is_raised_if_initial_sample_and_convex_body_have_incompatible_dims(self, sampler):
        with pytest.raises(ValueError):
            sampler(SQUARE, [0, 0, 0])

    def test_dim_returns_body_dimension(self, chain):
        assert chain.dim == 2

    def test_sample_throws_exception_if_n_is_negative(self, chain):
        with pytest.raises(ValueError):
            chain.sample(n=-1)

    def test_sample_throws_exception_if_n_is_zero(self, chain):
        with pytest.raises(ValueError):
            chain.sample(n=0)

    def test_sample_throws_exception_if_warmup_is_negative(self, chain):
        with pytest.raises(ValueError):
            chain.sample(warmup=-1)

    def test_sample_throws_exception_if_thin_is_negative(self, chain):
        with pytest.raises(ValueError):
            chain.sample(thin=-1)

    def test_sample_throws_exception_if_thin_is_zero(self, chain):
        with pytest.raises(ValueError):
            chain.sample(thin=0)

    def test_sample_throws_exception_if_chains_is_negative(self, chain):
        with pytest.raises(ValueError):
            chain.sample(chains=-1)

    def test_sample_throws_exception_if_chains_is_zero(self, chain):
        with pytest.raises(ValueError):
            chain.sample(chains=0)

    def test_warmup_and_thin(self, chain):
        all_samples = chain.sample(n=20, random_state=0)
        np.testing.assert_allclose(
            chain.sample(n=3, warmup=5, thin=4, random_state=0),
            all_samples[5:17:4]
        )

    def test_equal_seeds_return_identical_chains(self, chain):
        np.testing.assert_allclose(
            chain.sample(5, random_state=1),
            chain.sample(5, random_state=1)
        )

    def test_distinct_seeds_return_distinct_chains(self, chain):
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
