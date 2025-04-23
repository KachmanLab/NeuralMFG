from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pylab
from matplotlib.ticker import MultipleLocator

from games.meeting_arrival_times.MeetingTime import MeetingTime

import jax.random as jrandom
from diffrax import diffeqsolve, ControlTerm, Euler, MultiTerm, ODETerm, SaveAt, VirtualBrownianTree

plt.rcParams['font.size'] = 11


def main(suppress: bool = False):
    np.random.seed(10)

    mu = 12
    sigma = 0.5

    N = 6
    M = 7

    threshold = 0.8

    # Generate initial distribution
    initial_distribution = np.random.normal(mu, sigma, size=N * M)

    # Initialize meeting
    t0 = 1
    t1 = 16
    meeting = MeetingTime(mu, threshold)
    brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3, shape=(), key=jrandom.PRNGKey(0))
    terms = MultiTerm(ODETerm(meeting.drift), ControlTerm(meeting.diffusion, brownian_motion))

    solver = Euler()
    save_range = np.arange(t0, t1, 1)
    saveat = SaveAt(ts=save_range)
    sol = diffeqsolve(terms, solver, t0, t1, dt0=0.05, y0=initial_distribution, saveat=saveat)

    if not suppress:
        game_boxplots(sol.ys, r'./games/meeting_arrival_times/figures/arrival_times_example.pdf', left=True,
                      legend=False)


def game_boxplots(ys: np.ndarray, path: str, data_mean: float = None, left: bool = True, legend: bool = True):
    """
    Generate boxplots showing the development of the distribution throughout the game.

    Args:
        ys (np.ndarray): The distribution of players at each turn of the game.
        path (str): The path to save the figure.
        data_mean (float): The mean of the distribution where the data are drown from.
        left (bool): True if the figure is placed to the left, requiring a description for the y-axis. Default = True.
        legend (bool): True to include a legend in the figure. Default = True.

    Returns:

    """
    fig, axes = plt.subplots(1, 1, figsize=(5, 2.5))
    axes.boxplot(ys.T)

    if data_mean is not None:
        axes.axhline(data_mean, linewidth=0.8, label=r"$\mu_{data}$", color='tab:red')

    axes.set_xlabel("Turn index ($t$)")
    if left:
        axes.set_ylabel(r"Arrival time ($\tilde{\tau}$)")


    # Inline legend
    h, l = axes.get_legend_handles_labels()
    if legend:
        axes.legend(handles=h, labels=l, loc='lower left')

    # Ticks
    axes.yaxis.set_minor_locator(MultipleLocator(0.5))
    yticks = np.arange(axes.get_yticks()[0], axes.get_yticks()[-1], 1)
    if yticks.shape[0] > 8:
        yticks = yticks[::2]
    axes.yaxis.set_ticks(yticks)
    axes.yaxis.set_ticklabels([f"{int(tick)}:00" for tick in axes.get_yticks()])

    axes.xaxis.set_major_locator(MultipleLocator(2))
    axes.xaxis.set_major_formatter('{x:.0f}')
    axes.xaxis.set_minor_locator(MultipleLocator(1))

    if path is not None:
        fig.tight_layout()
        Path('/'.join(path.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
    else:
        plt.show()

    # Save legend
    if path is not None:
        figlegend = pylab.figure()
        n_cols = 1 if data_mean is None else 2
        figlegend.legend(h, l, ncol=n_cols, loc='center')
        figlegend.tight_layout()
        figlegend.savefig('/'.join(path.split('/')[:-1]) + '/arrival_times_legend.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
