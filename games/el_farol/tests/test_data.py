import numpy as np
from games.el_farol.bar_data import generate_probabilities


def test_data_generation():
    N = 600  # Number of distributions to sample from
    M = 7  # Number of samples per distribution

    mu = 0.5  # The official starting time of the meeting
    sigma = 0.1  # Starting time noise (perhaps due to incomplete communication)

    k = 2
    theta = 0.2

    _, _, p = generate_probabilities(mu, sigma, k, theta, N, M)

    assert p.shape == (N * M,)
    assert np.all((0 <= p) & ( p<= 1))
