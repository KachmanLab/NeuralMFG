import jax
import numpy as np

import pandas as pd

from tqdm import tqdm

import jax.random as jrandom
import jax.numpy as jnp

from diffrax import SaveAt, Solution


from games.liars_dice import dice_game_nODE, dice_game_example
from games.liars_dice.dice_game_data import generate_nODE_data, generate_opening_bid
from games.liars_dice.neuralODE import MFGModel
from games.liars_dice.DiceGame.DiceGameAnalyses import game_length
from games.liars_dice.DiceGame import DiscreteDiceGame


def main(lam: float, neural: bool):
    """
    Analyze the game using initial bluffing strategy `lam`.

    Args:
        lam (float): The initial bluffing strategy.
        neural (bool): True if the neural ODE should be analyzed.

    Returns:

    """
    # Model parameters
    n_games = 100
    model_params = {'n_players': 30,  # 10
                    'n_dice': 15,  # 5
                    'n_faces': 6,
                    'y_true': np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6, ]), # np.array([0.1, 0.2, 0.0, 0.4, 0.1, 0.2, ]),
                    'key': jrandom.PRNGKey(42),
                    'seed': 10,
                    'batch_size': 64,
                    'lam': lam
                    }

    # Models
    if neural:
        model = dice_game_nODE.main(300, True, **model_params, reduce_turns=False, dataset_size=100)
    else:
        model = dice_game_example.main(True, **model_params)

    # Game data
    games = generate_nODE_data(n_games, model_params['n_players'], model_params['n_dice'], model_params['y_true'],
                               model_params['n_faces'])
    # Initial carry
    carry0 = generate_opening_bid(model_params['n_faces'], n_players=model_params['n_players'],
                                  dice_odds=jnp.array(model_params['y_true']), lam=model_params['lam'])

    # Calculate the game length
    return analyze_dice_game(model, games, carry0, model_params['key'], max_save=100, print_info=True)




def challenge_status(carry: jax.Array):
    """
    Calculate the ratio of successful challenges from the `carry`.

    Args:
        carry (jax.Array): The parameters of the categorical distribution describing the bids.
                Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
                The second and third arrays are padded with zeroes to match the shape of the first array.
    Returns:
        float: the ratio of successful challenges
    """
    game_states = 0
    for game in carry:
        for turn in game:
            _, _, game_state = DiscreteDiceGame.unwrap_carry(turn.squeeze(), 0)
            if jnp.sum(game_state) > 0 and jnp.sum(game_state) != jnp.inf:
                game_states += 1
                break
    return game_states / carry.shape[0]


def average_bluff(carry: jax.Array) :
    """
    Calculate the average value of the bluff parameter lambda from the `carry`.

    Args:
        carry (jax.Array): The parameters of the categorical distribution describing the bids.
                Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
                The second and third arrays are padded with zeroes to match the shape of the first array.
    Returns:
        float: the average value of the bluff parameter.
    """
    n_players = carry.shape[1] - 2

    players_idx = jnp.arange(0, n_players, 1)
    for i, turn in enumerate(carry):
        y, _, game_state = DiscreteDiceGame.unwrap_carry(turn.squeeze(), players_idx)
        if jnp.sum(game_state) != 0 and jnp.sum(game_state) != jnp.inf:
            return jnp.mean(y[..., -1])
    return 0


def analyze_dice_game(model: MFGModel, games: jax.Array, carry0: jax.Array,
                      key: jrandom.PRNGKey, max_save: int = 1000, print_info: bool = False):
    """
    Calculate the average game length over `games`.

    Args:
        model (MFGModel): The game model.
        games (jax.Array): Game data.
        carry0 (jax.Array): The initial carry.
            Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
            The second and third arrays are padded with zeroes to match the shape of the first array.
        key (jrandom.PRNGKey): A random key.
        max_save (int): The maximal length of the output carry.
        print_info (bool): Prints the average length and standard deviation if True. Default = False.

    Returns:
        float, float, float, float, float:  The mean length of the games and its standard deviation.
                                            The mean bluff parameter and its standard deviation
                                            The challenge ratio
    """
    keys = jrandom.split(key, games.shape[0])
    saveat = SaveAt(ts=jnp.arange(0, max_save, 1))
    outcomes = jax.vmap(model, in_axes=(None, 0, None, 0))(carry0, games, saveat, keys)

    if type(outcomes) == Solution:
        outcomes = outcomes.ys

    lengths = jnp.array([game_length(outcome) for outcome in outcomes])
    bluffs = jnp.array([average_bluff(outcome) for outcome in outcomes])
    challenges = challenge_status(outcomes)


    mean, std = (jnp.mean(lengths), jnp.std(lengths))
    mean_l, std_l = (jnp.mean(bluffs), jnp.std(bluffs))

    if print_info:
        print(model)
        print(f"\tMean game length: {mean:.2f} turns.")
        print(f"\tStandard deviation: {std:.2f} turns.")
        print()
        print(f"\tMean $\lambda$: {mean_l:.3f}.")
        print(f"\tStandard deviation: {std_l:.3f}.")
        print()
        print(f"\tRatio of successful challenges: {challenges:.3f}.")
        print()

    return np.array([mean, std, mean_l, std_l, challenges])


if __name__ == '__main__':
    neural_ODE = True
    lambdas = np.arange(0, 31, 1)
    columns = ['lambda', 'length_mu', 'length_s', 'bluff_mu', 'bluff_s', 'challenge_ratio']

    results = np.zeros((lambdas.shape[0], len(columns)))

    for i, l in tqdm(enumerate(lambdas), total=lambdas.shape[0]):
        results[i, :] = np.append(np.array([l]), main(l, neural=neural_ODE))

    df = pd.DataFrame(results, columns=columns)
    df.to_csv(r'./games/liars_dice/analyses/game_analyses_nODE_100.csv')

