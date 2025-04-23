import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams['font.size'] = 11

def plot_analyses(path: str, l_max: int = 20, suffix: str = ""):
    """
    Plot the analyses found at `path`. Optionally, the path can be extended with `suffix` and the plot can be limited
    until `l_max`.

    Args:
        path (str): The path to the data.
        l_max (int): The maximal value of lambda to show.
        suffix (str): An additional suffix to add to the path.

    Returns:
        None.
    """
    standard = path + f'{suffix}.csv'
    nODE = path + f'_nODE{suffix}.csv'

    df_standard = pd.read_csv(standard, index_col=0)[:l_max]
    df_nODE = pd.read_csv(nODE, index_col=0)[:l_max]


    # Plot 1
    fig, ax = basic_plot(df_nODE, df_standard, 'lambda', 'length_mu')
    # axes formatting
    ax.set_xlabel("Initial bluff strategy ($\lambda_0$)")
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_major_formatter('{x:.1f}')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlim(-0.5, l_max + 0.5)
    ax.set_ylabel("Game length (turns)")
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.yaxis.set_minor_locator(MultipleLocator(2))
    fig.tight_layout()
    fig.savefig(fr'./games/liars_dice/analyses/figures/initial_bluff_vs_game_length{suffix}.pdf')


    # Plot 2
    fig, ax = basic_plot(df_nODE, df_standard, 'lambda', 'bluff_mu')
    # axes formatting
    ax.set_xlabel("Initial bluff strategy ($\lambda_0$)")
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_major_formatter('{x:.1f}')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlim(-0.5, l_max + 0.5)
    ax.set_ylabel("Learned bluff strategy ($\lambda$)")
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    fig.tight_layout()
    fig.savefig(fr'./games/liars_dice/analyses/figures/initial_bluff_vs_learned_bluff{suffix}.pdf')


    # Plot 3
    fig, ax = basic_plot(df_nODE, df_standard, 'lambda', 'challenge_ratio')
    # axes formatting
    ax.set_xlabel("Initial bluff strategy ($\lambda_0$)")
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_major_formatter('{x:.1f}')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlim(-0.5, l_max + 0.5)
    ax.set_ylabel("Ratio of successful\nchallenges")
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    fig.tight_layout()
    fig.savefig(fr'./games/liars_dice/analyses/figures/initial_bluff_vs_challenge_ratio{suffix}.pdf')


def basic_plot(df_nODE: pd.DataFrame, df_standard: pd.DataFrame, x_col: str, y_col: str):
    """
    Generate a basic plot given the two dataframes, the x_col and y_col.
    Args:
        df_nODE (pd.DataFrame): A pandas dataframe containing the analyses with the nODE.
        df_standard (pd.DataFrame): A pandas dataframe containing the analyses with the MFG.
        x_col (str): The column to plot on the x-axis.
        y_col (str): The column to plot on the y-axis.

    Returns:

    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
    ax.plot(df_standard[x_col], df_standard[y_col], '-s', label='MFG dynamics', markersize=4, color='tab:blue')
    ax.plot(df_nODE[x_col], df_nODE[y_col], '-o', label='neural ODE', markersize=4, color='tab:orange')
    ax.legend()
    return fig, ax


if __name__ == '__main__':
    plot_analyses(path=r'./games/liars_dice/analyses/csvs/game_analyses', l_max=11, suffix='_unfair')