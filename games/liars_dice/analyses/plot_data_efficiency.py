import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams['font.size'] = 11

def plot_efficiency(path: str):
    """
    Plot the results found at `path`.

    Args:
        path (str): The path to the data.

    Returns:
        None.
    """
    df = pd.read_csv(path, index_col=0).to_numpy()

    n_dice = int(re.search("\d+", path)[0])
    x = np.arange(1, df.shape[0] + 1) * n_dice

    fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))

    ax.errorbar(x, np.mean(df, axis = -1), yerr=np.std(df, axis = -1), capsize=2)

    # axes formatting
    ax.set_xlabel("Number of training dice")
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_ylabel(r"$KL(\theta, \hat{\theta}_T)$")
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_formatter('{x:.1f}')
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    fig.tight_layout()
    fig.savefig(fr'./games/liars_dice/analyses/figures/data_efficiency_{n_dice}_dice.pdf')

if __name__ == '__main__':
    plot_efficiency(path=r'./games/liars_dice/analyses/csvs/data_efficiency_4_dice.csv',)