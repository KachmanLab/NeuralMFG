import pytest

import numpy as np
from games.liars_dice.DiceGame.probability import at_least_q, exactly_q, jit_at_least_q, jit_calculate_loglikelihood

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def test_exactly():
    assert np.round(exactly_q(1, 6, 1 / 6).item(), 7) == 0.4018776
    assert np.round(exactly_q(2, 6, 1 / 6).item(), 7) == 0.2009388
    assert np.round(exactly_q(3, 6, 1 / 6).item(), 7) == 0.0535837
    assert np.round(exactly_q(4, 6, 1 / 6).item(), 7) == 0.0080376
    assert np.round(exactly_q(5, 6, 1 / 6).item(), 7) == 0.0006430
    assert np.round(exactly_q(6, 6, 1 / 6).item(), 7) == 0.0000214

    assert np.round(exactly_q(1, 5, 1 / 12).item(), 7) == 0.2941945
    assert np.round(exactly_q(2, 5, 1 / 12).item(), 7) == 0.0534899
    assert np.round(exactly_q(3, 5, 1 / 12).item(), 7) == 0.0048627
    assert np.round(exactly_q(4, 5, 1 / 12).item(), 7) == 0.0002210
    assert np.round(exactly_q(5, 5, 1 / 12).item(), 7) == 0.0000040


def test_at_least():
    assert np.round(at_least_q(1, 5, 1 / 6).item(), 6) == 0.598122
    assert np.round(at_least_q(2, 5, 1 / 6).item(), 6) == 0.196245
    assert np.round(at_least_q(3, 5, 1 / 6).item(), 6) == 0.035494
    assert np.round(at_least_q(4, 5, 1 / 6).item(), 6) == 0.003344

    assert np.round(at_least_q(1, 5, 1 / 12).item(), 4) == 0.3528
    assert np.round(at_least_q(2, 5, 1 / 12).item(), 4) == 0.0586
    assert np.round(at_least_q(3, 5, 1 / 12).item(), 4) == 0.0051
    assert np.round(at_least_q(4, 5, 1 / 12).item(), 4) == 0.0002


@pytest.mark.parametrize("q, n, p, out", [(1, 5, 1 / 6, 0.5981), (2, 5, 1 / 6, 0.1962), (3, 5, 1 / 6, 0.0355),
                                          (4, 5, 1 / 6, 0.0033), (1, 5, 1 / 12, 0.3528), (2, 5, 1 / 12, 0.0586),
                                          (3, 5, 1 / 12, 0.0051), (4, 5, 1 / 12, 0.0002)])
def test_jit_at_least(q, n, p, out):
    assert np.round(jit_at_least_q(q, n, p).item(), 4) == out


@pytest.mark.parametrize("throw, p, out", [(jnp.array([6, 6, 6, 6, 6, 6]),
                                            jnp.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]),
                                            np.power(1/6, 6)),
                                           (jnp.array([6, 6, 6, 6, 6, 6]),
                                            jnp.array([0, 0, 0, 0, 0, 1]),
                                            np.power(1, 6))
                                           ])
def test_jit_calculate_likelihood(throw, p, out):
    decimals = 4
    assert np.round(jit_calculate_loglikelihood(throw, p).item(), decimals) == np.round(np.log(out), decimals)
