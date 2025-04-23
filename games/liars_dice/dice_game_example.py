import numpy as np

import jax.numpy as jnp
import jax.random as jrandom

from diffrax import Euler, SaveAt

from games.liars_dice.dice_game_data import generate_nODE_data, generate_opening_bid
from games.liars_dice.neuralODE import MeanFieldGame
from games.liars_dice.DiceGame import analyse_nODE_solution, DiscreteDiceGame


def main(suppress: bool = False, **kwargs):
    # Random state management
    key = jrandom.PRNGKey(42) if 'key' not in kwargs else kwargs['key']
    seed = 10 if 'seed' not in kwargs else kwargs['seed']
    np.random.seed(seed)

    # Game parameters, optional via kwargs
    n_players = 7 if 'n_players' not in kwargs else kwargs['n_players']
    n_dice = 6 if 'n_dice' not in kwargs else kwargs['n_dice']
    n_faces = 6 if 'n_faces' not in kwargs else kwargs['n_faces']

    # Generate data
    # 1: Target distribution
    y_true = np.array([0.0, 0.1, 0.2, 0.4, 0.1, 0.2, ])

    # 2: Labels
    dice_outcomes = generate_nODE_data(1, n_players, n_dice, y_true)[0]

    # 3: Initial carry
    lam = 0.1 if 'lam' not in kwargs else kwargs['lam']
    carry0 = generate_opening_bid(n_faces, n_players, lam=lam)

    # Initialize game
    game = DiscreteDiceGame(n_players, n_dice, n_faces, debug=False)
    model = MeanFieldGame(game.drift, solver=Euler())
    saveat = SaveAt(ts=jnp.arange(0, 20, 1))
    sol = model(carry0, dice_outcomes, saveat, key)

    if not suppress:
        analyse_nODE_solution(sol, n_players)

    return model


if __name__ == '__main__':
    main()
