import diffrax
import jax
import numpy as np

from games.liars_dice.DiceGame.GameVariants import DiscreteDiceGame
from games.liars_dice.dice_game_data import generate_data

import jax.numpy as jnp
import jax.random as jrandom
from diffrax import ODETerm, Euler


# TODO: Allow players to "cheat" by re-rolling their dice
# In the paper: you can't have an analytical solution, but you can easily solve this with MFGs
# Leave the analytical solution as an exercise for the reader

# No easy tractable formulation for an analytical problem, it's hard to formulate this in a way that can be solved
# Later: Calculus of variations approach -> learn the formulation

def main():
    # Parameters
    n_players = 7
    n_dice = 6
    n_faces = 6

    # Generate an opening bid
    dice_odds = jnp.ones(n_faces)/n_faces
    initial_bluff_lam = 0.3
    distribution_parameters = jnp.append(dice_odds, initial_bluff_lam)
    opening_bid = jnp.zeros_like(distribution_parameters)
    opening_bid = opening_bid.at[0].set(1)

    # Set the game_state
    state = jnp.zeros_like(distribution_parameters)

    carry0 = jnp.stack((distribution_parameters, opening_bid, state))

    # Target distribution
    y_true = np.array([0.0, 0.1, 0.2, 0.4, 0.1, 0.2, ])
    print(f"The sum of the target distribution is {np.sum(y_true)}")
    initial_outcomes = jnp.array(generate_data(n_players, n_dice, p=y_true))

    # Initialize game
    t0 = 0
    t1 = n_players * n_dice
    key = jrandom.PRNGKey(42)

    args = {'key': key, 'dice_outcomes': initial_outcomes, 'y_true': y_true}
    game = DiscreteDiceGame(n_players, n_dice, n_faces, debug=True)
    terms = ODETerm(game.drift)

    solver = Euler()
    play_game(carry0, t0, t1, n_players, solver, args, terms, True)


def play_game(carry0: jax.Array, t0: int, t1: int, n_players: int, solver: diffrax.AbstractSolver, args: dict,
              terms: diffrax.ODETerm, debug: bool = False):
    """
    Play a single game of Liar's dice.

    Args:
        carry0 (jax.Array): The initial carry. 
        t0 (int): The first turn of the game.
        t1 (int): The last turn of the game.
        n_players (int): The number of players participating in the game.
        solver (diffrax.AbstractSolver): The nODE solver.
        args (dict): A dictionary of game arguments.
        terms (diffrax.ODETerm): The game terms.
        debug (bool): True to print status statements.

    Returns:

    """
    turn_counter = 0
    tprev = t0
    dt0 = 1
    tnext = t0 + dt0
    y = carry0
    state = solver.init(terms, tprev, tnext, carry0, args)
    while tprev < t1:
        if turn_counter % n_players == 0 and debug:
            print(f"Round {turn_counter // n_players + 1}:")
        y, _, _, state, _ = solver.step(terms, tprev, tnext, y, args, state, made_jump=False)
        if jnp.sum(y[1, :-1]) <= 0:  # Manual challenge mechanism
            print(f"Player {tprev % n_players + 1} challenged the bid and ended the game.")
            game_state = y[2]
            if jnp.sum(game_state) > 0:
                if debug:
                    print(f"Player {tprev % n_players + 1} won the challenge.")
            else:
                if debug:
                    print(f"Player {tprev % n_players + 1} lost the challenge.")
            break
        tprev = tnext
        tnext = min(tprev + dt0, t1)
        turn_counter += 1


if __name__ == '__main__':
    main()
