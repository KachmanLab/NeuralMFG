from typing import Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def main():
    N = 7  # Number of players
    D = 6  # Number of dice per player

    rolled_outcomes = generate_data(N, D)

    plot_outcomes(rolled_outcomes)

    return rolled_outcomes


def generate_data(N: int, D: int, p: np.ndarray = None, max_pips: int = 6) -> np.ndarray:
    """
    Generate a joint dice roll with `N` players throwing `D` dice each.

    Args:
        N (int): The number of players.
        D (int): The number of dice per player.
        p (np.ndarray): The probabilities associated with each die face.
            If not given, the sample assumes a uniform distribution over all entries.
        max_pips (int): The maximal number of pips on a die (default = 6, for we're not playing D&D by default here)

    Returns:
        np.ndarray: The dice outcomes.
    """
    assert N > 0, f"The number of players should be larger than 0, received {N}."
    assert D > 0, f"The number of dice per player should be larger than 0, received {D}."
    die = np.arange(1, max_pips + 1, 1)  # The possible outcomes of a die
    rolled_outcomes = np.random.choice(die, (N, D), p=p)  # Draw the outcomes from a uniform distribution
    return rolled_outcomes


def plot_outcomes(outcomes: np.ndarray) -> None:
    """
    Plot the outcomes of the joint dice roll.
    Args:
        outcomes (np.ndarray): The dice outcomes of the shape (players, dice).

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 1)

    for index, player in enumerate(outcomes):
        unique, counts = np.unique(player, return_counts=True)
        for u, c in zip(unique, counts):
            ax.scatter(index, u, s=20 * c, color=colors[index])
            ax.annotate(c, (index, u))

    ax.set_xlabel("Player index")
    ax.set_ylabel("Dice outcome")

    fig.tight_layout()
    Path("./games/liars_dice/figures/").mkdir(parents=True, exist_ok=True)
    plt.savefig(r'./games/liars_dice/figures/dice_game_data.pdf')


def generate_opening_bid(n_faces: int, n_players: int, dice_odds: Union[jax.Array, None] = None, lam: float = 0.3):
    """
    Generate an opening bid given the number of faces the die has.

    Args:
        n_faces (int): The number of faces of the die.
        n_players (int): The number of players.
        dice_odds (jax.Array): The initial distribution of dice probabilities.
        lam (float): The initial bluff probability (parameterizes a poisson distribution).

    Returns:
        jax.Array: A carry containing
                        The distribution parameters, using a uniform prior. Also contains the bluff probability.
                        An opening bid, which is the lowest possible bid, an array of zeros with a 1 on the first index.
                        The game state, a vector of zeroes.

    """
    # Generate an opening bid
    dice_odds = jnp.ones(n_faces) / n_faces if dice_odds is None else dice_odds
    distribution_parameters = jnp.append(dice_odds, lam)

    opening_bid = jnp.zeros_like(distribution_parameters)
    opening_bid = opening_bid.at[0].set(1)
    opening_bid = opening_bid[jnp.newaxis, :]

    # Set the game_state
    state = jnp.zeros_like(distribution_parameters)[jnp.newaxis, :]

    # Broadcast the distribution parameters for all players
    distribution_parameters = jnp.repeat(distribution_parameters[jnp.newaxis], n_players, axis=0)

    carry0 = jnp.vstack((opening_bid, state, distribution_parameters))
    return carry0


def generate_nODE_data(S: int, N: int, D: int, y_true: np.ndarray, P: int = 6) -> jax.Array:
    """
    Generate nODE data. the generated dataset contains D observations of N * D dice throws. Final shape: (S, N, D). The
    dice throws are drawn according to the probabilities described in `y_true`.

    Args:
        S (int): The number of datapoints.
        N (int): The number of players.
        D (int): The number of dice per player.
        y_true (np.ndarray): The probabilities of throwing each dice outcome, should sum to 1.
        P (int): The number of faces per die.

    Returns:
        jax.Array: The data in shape (S, N, D).
    """
    assert np.round(np.sum(y_true), 3) == 1.0, (f"The sum of the target distribution should be 1.0,"
                                                f" received {np.sum(y_true)}")
    ys = jnp.array([generate_data(N, D, p=y_true, max_pips=P) for _ in range(S)])  # [s, n_players, n_dice]
    return ys


if __name__ == '__main__':
    main()
