import jax
import numpy as np
import pandas as pd
import scipy.special

import jax.random as jrandom
import jax.numpy as jnp

from diffrax import SaveAt


from games.liars_dice import dice_game_nODE
from games.liars_dice.dice_game_data import generate_nODE_data, generate_opening_bid

def efficiency_analyses(dataset_size: int, validation_size: int):
    """
    Train and analyze the model using `dataset_size` and `validation_size`.
    Args:
        dataset_size (int): The size of the training dataset.
        validation_size (int): The size of the validation dataset.

    Returns:
        The KL-divergence between the true and estimated dice odds.
    """
    # Model parameters
    n_games = validation_size
    model_params = {'n_players': 2,
                    'n_dice': 1,
                    'n_faces': 6,
                    'y_true': np.array([0.1, 0.2, 0.0, 0.4, 0.1, 0.2, ]),
                    'key': jrandom.PRNGKey(42),
                    'seed': 10,
                    'batch_size': 1,
                    'lam': 0.1
                    }

    # Train model
    nODE = dice_game_nODE.main(200, True, **model_params, reduce_turns=False, dataset_size=dataset_size)

    # Game data
    games = generate_nODE_data(n_games, model_params['n_players'], model_params['n_dice'], model_params['y_true'],
                               model_params['n_faces'])
    # Initial carry
    carry0 = generate_opening_bid(model_params['n_faces'], n_players=model_params['n_players'],
                                  dice_odds=jnp.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6]), lam=model_params['lam'])

    # Calculate outcomes
    keys = jrandom.split(model_params['key'], games.shape[0])
    saveat = SaveAt(t1=True)
    outcomes = jax.vmap(nODE, in_axes=(None, 0, None, 0))(carry0, games, saveat, keys)

    ys = outcomes.ys
    theta = np.array(ys[:, 0, 2:, :-1])

    probabilities = scipy.special.softmax(theta, axis=-1)

    # Average over the players
    probabilities = np.mean(probabilities, axis=1)

    # Calculate test KL
    return np.sum(np.array([scipy.special.kl_div(model_params['y_true'], p) for p in probabilities]), axis=-1)

if __name__ == '__main__':
    validation_size = 1000
    sizes = np.arange(1, 20, 1, dtype=int)
    divergences = np.zeros((sizes.shape[0], validation_size))

    for i, size in enumerate(sizes):
        divergences[i, :] = efficiency_analyses(size, validation_size)
        print(f'Dataset size: {size}, mean KL-divergence: {np.mean((divergences[i])):.3f}, std: {np.std(divergences[i]):.4f}')

    df = pd.DataFrame(divergences)
    df.to_csv(r'./games/liars_dice/analyses/csvs/data_efficiency_2_dice.csv')

