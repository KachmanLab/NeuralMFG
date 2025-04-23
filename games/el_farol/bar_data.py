from typing import Union
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from games.meeting_arrival_times.arrival_times_data import generate_arrival_times

import jax
import jax.numpy as jnp
import jax.random as jrandom

# TODO: Generate data using poisson distribution


def main():
    N = 6  # Number of distributions to sample from
    M = 7  # Number of samples per distribution

    mu = 0.3  # The mean intention of going to the bar
    sigma = 0.01  # Noise around this intention

    k = 2
    theta = 0.01

    mu, sigma, p = generate_probabilities(mu, sigma, k, theta, N, M)

    plot_bar_probabilities(p.reshape(N, M))


def generate_probabilities(mu: float, sigma: float, k: float, theta: float, N: int, M: int):
    """
    Generate $N \times M$ probabilities of going to the bar. The probability of going to the bar is drawn from a
    Gaussian distribution with mean `mu` (the intention of going to the bar) and scale `sigma` and a Gamma distribution
     with shape `k` and scale `theta`. The probabilities of going to the bar are normalized between 0 and 1.

     Uses the function `generate_arrival_times` under the hood.

    Args:
        mu (float): The mean of the Gaussian distribution used to generate means.
        sigma (float): The scale of the Gaussian distribution used to generate means.
        k (float): The shape of the Gamma distribution used to generate scales.
        theta (float): The shape of the Gamma distribution used to generate scales.
        N (int): The number of distributions to generate.
        M (int): The number of samples drawn per distribution.

    Returns:
        np.ndarray: The means of the intermediate distributions, the intentions (shape N,).
        np.ndarray: The standard deviations of the intermediate distributions, the peer pressure (shape N,).
        np.ndarray: The probabilities of going to the bar, a distribution of (Shape NxM,).
    """
    mu_i, sigma_i, x_j = generate_arrival_times(mu, sigma, k, theta, N, M)
    return mu_i, sigma_i, cap(x_j)


def cap(a: np.ndarray, minimum: float = 0, maximum: float = 1) -> Union[np.ndarray, jax.Array]:
    """
    Cap `a` between `minimum` and `maximum`.

    Args:
        a (np.ndarray): An numpy array.
        minimum (float): The minimal required value.
        maximum (float): The maximal required value.

    Returns:
        np.array: The capped array.
    """
    a = jnp.minimum(a, maximum)
    a = jnp.maximum(a, minimum)
    return a


def plot_bar_probabilities(p: np.ndarray) -> None:
    """
    Generate boxplots of the distribution `p`.

    Args:
        p (np.ndarray): A numpy array.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 1)
    axes.boxplot(p)
    axes.set_ylabel("The probability of going to the bar")
    axes.set_xlabel("Subgroup")

    fig.tight_layout()
    Path("./games/el_farol/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig(r'./games/el_farol/figures/bar_data.pdf')


def normalize(a: np.ndarray):
    """
    Scale the values in `a` between 0 and 1.

    Args:
        a (np.ndarray): A numpy array.

    Returns:
        np.ndarray: The variant of a where all entries are scaled between 0 and 1.
    """
    return (a - np.min(a)) / (np.max(a) - np.min(a))


def probabilities_to_attendance_rates(p: Union[np.ndarray, jax.Array], n: int, key: jrandom.PRNGKey) \
        -> Union[np.ndarray, jax.Array]:
    """
    Draw attendance rates from the vector of probabilities `p`.

    Args:
        p (np.ndarray): The vector of probabilities, shape = (n_datapoints, n_agents,).
        n (int): The number of observations per vector.
        key (jrandom.PRNGKey): A random key.

    Returns:
        np.ndarray: The vector of attendance rates, shape = (n_datapoints, ).
    """
    keys = jrandom.split(key, n)
    return jnp.array([jnp.sum(jrandom.bernoulli(k, p=cap(p)), axis=-1) for k in keys]).T


if __name__ == '__main__':
    main()
