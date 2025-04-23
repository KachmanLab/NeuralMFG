import numpy as np

import jax
import jax.numpy as jnp

from functools import partial
from games.liars_dice.DiceGame.DiceGame import DiceGame


def analyse_nODE_solution(carry: jax.Array, n_players: int) -> None:
    """
    Analyse the neural ODE solution, unwrapping the carry into bids, turns and challenges.

    Args:
        carry (jax.Array): The parameters of the categorical distribution describing the bids.
                Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
                The second and third arrays are padded with zeroes to match the shape of the first array.
        n_players (int): The number of players.

    Returns:
        None
    """
    np.set_printoptions(precision=3, suppress=True)
    turn_counter = 1
    for i, player in enumerate(carry):
        player_idx = i % n_players
        if player_idx == 0:
            print(f"Turn: {turn_counter}")
            turn_counter += 1

        y, previous_bid, game_state = DiceGame.unwrap_carry(player.squeeze(), player_idx)
        if jnp.sum(game_state) > 0:
            print(f"\tPlayer {player_idx + 1} successfully challenged the bid. (Game state = {game_state})")
            break
        elif jnp.sum(game_state) < 0:
            print(f"\tPlayer {player_idx + 1} unsuccessfully challenged the bid. (Game state = {game_state})")
            break

        print(f"\tPlayer {player_idx + 1} bids: {np.array(previous_bid)}")
        print(f"\t\tEstimated dice probabilities: {np.array(y)[:-1]}, sum: {np.sum(np.array(y[:-1])):.2f}")
        print(f"\t\tBluff parameter: {np.array(y)[-1]:.3f}")
        print(f"\t\tGame state: {game_state}")

    print("The game has ended.")


def game_length(carry: jax.Array) -> int:
    """
    Calculate the game length in turns from the `carry`.

    Args:
        carry (jax.Array): The parameters of the categorical distribution describing the bids.
                Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
                The second and third arrays are padded with zeroes to match the shape of the first array.
    Returns:
        int: The game length in turns.
    """
    for i, turn in enumerate(carry):
        _, _, game_state = DiceGame.unwrap_carry(turn.squeeze(), 0)
        if jnp.sum(game_state) != 0 and jnp.sum(game_state) != jnp.inf:
            return i
    return carry.shape[0]


def challenge_status(carry: jax.Array) -> int:
    """
    Get the challenge status from the `carry`.

    Args:
        carry (jax.Array): The parameters of the categorical distribution describing the bids.
                Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
                The second and third arrays are padded with zeroes to match the shape of the first array.
    Returns:
        int: The challenge status
    """
    for i, turn in enumerate(carry):
        _, _, game_state = DiceGame.unwrap_carry(turn.squeeze())
        if jnp.sum(game_state) != 0:
            return game_state.item()
    return 0


def game_parameters(carry: jax.Array) -> jax.Array:
    """
    Unwrap the distribution parameters from the `carry`.

    Args:
        carry (jax.Array): The parameters of the categorical distribution describing the bids.
                Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
                The second and third arrays are padded with zeroes to match the shape of the first array.
    Returns:
        int: The distribution parameters (n_turns, [*n_faces die odds, bluff_probability])
    """
    for i, turn in enumerate(carry):
        _, _, game_state = DiceGame.unwrap_carry(turn.squeeze())
        if jnp.sum(game_state) != 0:
            return carry[i, 0, :]
    return carry[-1, 0, :]
