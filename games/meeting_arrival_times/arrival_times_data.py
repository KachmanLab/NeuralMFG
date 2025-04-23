from typing import Tuple
from pathlib import Path

import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import patches

plt.rcParams['font.size'] = 11


def main():
    N = 6  # Number of distributions to sample from
    M = 7  # Number of samples per distribution

    mu = 12  # The official starting time of the meeting
    sigma = 0.5  # Starting time noise (perhaps due to incomplete communication)

    k = 2
    theta = 0.2

    mu_i, sigma_i, x_j = generate_arrival_times(mu, sigma, k, theta, N, M)
    plot_arrival_times(x_j, mu, sigma, mu_i, sigma_i)


def generate_arrival_times(mu: float, sigma: float, k: float, theta: float, N: int, M: int) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate $N \times M$ arrival times that are indirectly drawn from a Gaussian distribution with mean `mu` and scale
    `sigma` and a Gamma distribution with shape `k` and scale `theta`.

    Args:
        mu (float): The mean of the Gaussian distribution used to generate means.
        sigma (float): The scale of the Gaussian distribution used to generate means.
        k (float): The shape of the Gamma distribution used to generate scales.
        theta (float): The shape of the Gamma distribution used to generate scales.
        N (int): The number of distributions to generate.
        M (int): The number of samples drawn per distribution.

    Returns:
        np.ndarray: The means of the intermediate distributions (shape N,).
        np.ndarray: The standard deviations of the intermediate distributions (shape N,).
        np.ndarray: The arrival times (Shape NxM,).
    """
    mu_i = np.random.normal(mu, sigma, size=N)
    sigma_i = np.random.gamma(k, theta, size=N)
    x_j = np.random.multivariate_normal(mu_i, sigma_i * np.eye(N), M).reshape(N * M)
    return mu_i, sigma_i, x_j


def plot_arrival_times(x_j: np.ndarray, mu, sigma, mu_i: np.ndarray, sigma_i: np.ndarray, save: bool = True) -> None:
    """
    Plot a normalized histogram of the arrival times together with the distributions used to generate the samples.

    Args:
        x_j (np.ndarray): The arrival times (Shape NxM,).
        mu (float): The mean of the original distribution.
        sigma (float): The standard deviation of the original distribution.
        mu_i (np.ndarray): The means of the intermediate distributions (shape N,).
        sigma_i (np.ndarray): The standard deviations of the intermediate distributions (shape N,).
        save (bool): Save the plot to the disk if True, else, show the plot.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 1, figsize=(5, 4))

    # Histogram
    _, bins, _ = axes.hist(x_j, density=True, color='gray', alpha=0.6, )

    # Normal distributions
    data_range = np.linspace(bins[0], bins[-1], 1000)
    for m, s in zip(mu_i, sigma_i):
        axes.plot(data_range, norm.pdf(data_range, m, s) / mu_i.shape[0], linestyle='--')
    axes.plot(data_range, norm.pdf(data_range, mu, sigma), linestyle='-', color='black')

    # Create legend
    hist_marker = patches.Patch(color='gray', alpha=0.6, label='Observed arrival times')
    line_marker = Line2D([0], [0], linestyle='--', color='gray', label='Sampled normal\ndistributions (any color)')
    original_marker = Line2D([0], [0], linestyle='-', color='black', label='Original normal distribution')
    axes.legend(handles=[hist_marker, original_marker, line_marker], loc='upper left')

    # Axis markup
    axes.set_xlabel("Time")
    axes.set_ylabel("Density")
    axes.xaxis.set_minor_locator(MultipleLocator(0.5))
    axes.xaxis.set_major_locator(MultipleLocator(1))
    axes.xaxis.set_ticks(axes.get_xticks())
    axes.xaxis.set_ticklabels([f"{int(tick)}:00" for tick in axes.get_xticks()])

    if save:
        fig.tight_layout()
        Path("./games/meeting_arrival_times/figures").mkdir(parents=True, exist_ok=True)
        plt.savefig(r'./games/meeting_arrival_times/figures/arrival_times_data.pdf')
    else:
        plt.show()

if __name__ == '__main__':
    main()
