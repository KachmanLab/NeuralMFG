from typing import Union

import scipy
import numpy as np

import jax
import jax.numpy as jnp


def exactly_q(q: Union[int, jax.Array], n: Union[int, jax.Array], p: Union[float, jax.Array] = 1/6) \
        -> Union[int, jax.Array]:
    """Calculate the probability that at exactly `q` dice are showing a given face for a total of `n` dice with single
     outcome probability `p`."""
    return jax.scipy.stats.binom.pmf(q, n, p)


def at_least_q(q: Union[int, jax.Array], n: Union[int, jax.Array], p: float = 1/6) -> Union[int, jax.Array]:
    """Calculate the probability that at least `q` dice are showing a given face for a total of `n` dice with single
    outcome probability `p`."""
    def sum_function(i, x):
        return x + exactly_q(i, n, p)
    return jax.lax.fori_loop(q, n + 1, sum_function, 0)


def probability_lookup_table(n: Union[int, jax.Array], p: Union[float, jax.Array]):
    """Create a lookup table for the probability of throwing at least q in [0, `n`] outcomes with single outcome
     probability `p`."""
    q_range = jnp.arange(1, n + 1)
    probabilities = jax.vmap(exactly_q, in_axes=(0, None, None))(q_range, n, p)
    return jnp.cumsum(probabilities[::-1])[::-1]  # Reverse cumulative sum


def jit_at_least_q(q: Union[int, jax.Array], n: Union[int, jax.Array], p: Union[float, jax.Array]):
    """Calculate the probability that at least `q` dice are showing a given face for a total of `n` dice with single
    outcome probability `p`. (jit variant)."""
    return probability_lookup_table(n, p)[q-1]


def jit_theta_from_throw(dice_throw: jax.Array, n_faces: Union[int, jax.Array]) -> jax.Array:
    """
    Estimate the dice parameters from a single throw.

    Args:
        dice_throw (jax.Array): The dice outcomes.
        n_faces (int): The number of possible faces.

    Returns:
        jax.Array: The parameters that are estimated from the dice throw.
    """
    dice_throw = dice_throw.flatten()
    counts = jnp.bincount(dice_throw, length=n_faces + 1)[1:]  # The first index is associated with the face 0.
    return counts / dice_throw.shape[0]


def kullback_leibler(x: jax.Array, theta_pred: jax.Array, epsilon: float = 1e-5):
    """
    Calculate the KL divergence between the distribution estimated from the dice throw `x` and the predicted
    distribution `theta_pred`.

    Args:
        x (jax.Array): The dice throws associated with the carry.
        theta_pred (jax.Array): The parameters of the categorical distribution describing the bids.
                Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
                The second and third arrays are padded with zeroes to match the shape of the first array.
        epsilon (float): A float to prevent division by 0.

    Returns:
        jax.Array: The KL divergence between the predicted and estimated distribution parameters.
    """
    theta_estimated = jit_theta_from_throw(x, theta_pred.shape[0])
    KL = jax.scipy.special.kl_div(theta_estimated + epsilon, theta_pred + epsilon)
    return jnp.sum(KL)


def calculate_likelihood(dice_throw: np.ndarray, p: jnp.ndarray):
    """Calculate the loglikelihood of the outcomes in `dice_throw` given the dice odds `p`."""
    f, c = np.unique(dice_throw, return_counts=True)

    val = 0
    for f_i, c_i in zip(f, c):
        print(c_i, f_i, p[f_i-1])
        val += scipy.special.binom(c, dice_throw.shape) * exactly_q(c_i, c_i, p[f_i-1])

    return val


def jit_calculate_loglikelihood(dice_throw: jax.Array, p: Union[float, jax.Array]):
    """Calculate the loglikelihood of the outcomes in `dice_throw` given the dice odds `p`."""
    dice_throw = dice_throw.flatten()
    f, c = jnp.unique(dice_throw, return_counts=True, size=p.shape[0])

    def exactly_q_helper(i, x):
        idx = f[i]-1  # Get the index of the probability corresponding to face f[i]
        return x + jit_binom(c[i], dice_throw.shape[0]) * exactly_q(c[i], c[i], p[idx])

    return jnp.log(jax.lax.fori_loop(0, c.shape[0], exactly_q_helper, 0))


def jit_binom(x, y):
    """Jit version of the binomial coefficient"""
    return jnp.exp(jax.scipy.special.gammaln(x + 1) -
                   jax.scipy.special.gammaln(y + 1) -
                   jax.scipy.special.gammaln(x - y + 1))
