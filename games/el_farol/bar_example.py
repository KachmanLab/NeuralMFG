from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pylab
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from games.el_farol.BarProblem import BarProblem

import jax.numpy as jnp
from diffrax import diffeqsolve, Euler, ODETerm, SaveAt

plt.rcParams['font.size'] = 11


def main(suppress: bool = False):
    # Random state management
    np.random.seed(10)

    # Game parameters
    N = 6
    M = 7
    threshold = 0.9

    # Generate initial distribution
    initial_distribution = jnp.array(np.random.random(N * M))

    # Initialize the bar
    t0 = 1
    t1 = 16
    el_farol = BarProblem(threshold)
    terms = ODETerm(el_farol.drift)

    solver = Euler()
    save_range = np.arange(t0, t1, 1)
    saveat = SaveAt(ts=save_range)
    sol = diffeqsolve(terms, solver, t0, t1, dt0=0.05, y0=initial_distribution, saveat=saveat)

    if not suppress:
        bar_boxplots(sol.ys, threshold, r'./games/el_farol/figures/bar_example.pdf', left=True)

def bar_boxplots(ys: np.ndarray, threshold: float, path: str, data_mean: float = None, left: bool = True) -> None:
    """
    Generate boxplots showing the development of the distribution throughout the game.

    Args:
        ys (np.ndarray): The distribution of players at each turn of the game.
        threshold (float): The crowding threshold of the bar.
        path (str): The path to save the figure.
        data_mean (float): The mean of the distribution where the data are drown from.
        left (bool): True if the figure is placed to the left, requiring a description for the y-axis. Default = True.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5), sharey='all', gridspec_kw={'width_ratios': [5, 1]})

    axes[0].boxplot(ys.T)
    axes[0].axhline(threshold, linewidth=0.8, label="Crowding threshold")
    handles, labels = axes[0].get_legend_handles_labels()
    if data_mean is not None:
        axes[0].axhline(data_mean, linewidth=0.8, label=r"$\mu_{data}$", color='tab:red')
    else:
        data = Line2D([0], [0], label=r'$\mu_{data}$', color='tab:red', linewidth=0.8)
        handles.extend([data])

    axes[0].set_xlabel("Turn index ($t$)")
    if left:
        axes[0].set_ylabel("Probability of going\nto the bar ($p$)")

    # Set y ticks
    axes[0].yaxis.set_major_locator(MultipleLocator(0.25))
    axes[0].yaxis.set_major_formatter('{x:.2f}')

    # For the minor ticks, use no labels; default NullFormatter.
    axes[0].yaxis.set_minor_locator(MultipleLocator(0.05))

    axes[0].xaxis.set_major_locator(MultipleLocator(2))
    axes[0].xaxis.set_major_formatter('{x:.0f}')
    axes[0].xaxis.set_minor_locator(MultipleLocator(1))



    h, l = axes[0].get_legend_handles_labels()

    # Histogram
    bins = np.arange(0.0, 1.05, 0.05)
    axes[1].hist(ys[-1, :], orientation='horizontal', weights=np.ones(ys.shape[1])/ys.shape[1], bins=bins)
    axes[1].set_xlim(0, 1)

    # x-axis formatting
    axes[1].set_xlabel(f'Density\n at $t=15$')
    axes[1].xaxis.set_major_locator(MultipleLocator(0.5))
    # axes[1].xaxis.set_major_formatter('{x:.1f}')
    axes[1].set_xticks([0, 0.5, 1])
    axes[1].set_xticklabels(['0',' 0.5', '1'])
    axes[1].xaxis.set_minor_locator(MultipleLocator(0.1))

    # Inline legend
    if left:
        axes[0].legend(handles=handles, loc = 'lower right')
    fig.tight_layout()

    plt.subplots_adjust(wspace=0.1, hspace=0)


    Path('/'.join(path.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close(fig)

    # Save legend
    figlegend = pylab.figure()
    n_cols = 1 if data_mean is None else 2
    figlegend.legend(h, l, ncol=n_cols, loc='center')
    figlegend.tight_layout()
    figlegend.show()
    figlegend.savefig('/'.join(path.split('/')[:-1]) + '/el_farol_legend.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
