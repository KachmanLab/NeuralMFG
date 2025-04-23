from typing import Tuple, Union

import diffrax

import jax
import jax.numpy as jnp
import jax.random as jrandom

import equinox as eqx
import optax.losses

from games.liars_dice.DiceGame import DiscreteDiceGame
from games.liars_dice.neuralODE.NeuralODE import NeuralODE
from games.liars_dice.DiceGame.probability import jit_calculate_loglikelihood, kullback_leibler, jit_theta_from_throw
# TODO: Add analyses when length loss is incorporated, yes or no


@eqx.filter_value_and_grad
def grad_loss(model: NeuralODE, carry0: jax.Array, yi: jax.Array, key: jrandom.PRNGKey) \
        -> jax.Array:
    """
    Calculate the loss and gradients of the liar's dice game. Uses `jax`' vmap for efficiency.

    Args:
        model (NeuralODE): The neural ODE model.
        carry0 (jax.Array): The initial carry.
            Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
            The second and third arrays are padded with zeroes to match the shape of the first array.
        yi (jax.Array): A collection of dice throws (shape: (size, n_players, n_dice)).
        key (jrandom.PRNGKey): A random key.

    Returns:
        Tuple[jax.Array, jax.Array]: The loss and the gradients of the liar's dice game.
    """
    # Broadcast the key to prepare the vmap
    keys = jrandom.split(key, yi.shape[0])

    # Make batched predictions vmap!
    y_sol = jax.vmap(model, in_axes=(None, 0, None, 0))(carry0, yi, diffrax.SaveAt(t1=True), keys)
    y_pred = jax.vmap(lambda s: s.ys, in_axes=0)(y_sol)

    # Add a penalty for bluffing too much -> too many games are challenged correctly
    bluff_loss = MSE_bluff_loss(y_pred[:, -1, :, :])

    # Distribution loss: another vmap!
    distribution_loss = jnp.mean(jax.vmap(calculate_KL_loss, in_axes=(0, 0))(yi, y_pred[:, -1, :, :]))

    # Combine the losses
    loss = distribution_loss + 0.8 * bluff_loss

    if model.reduced_turns:
        # Return loss -> finish the game early to prevent that we will be called for another round
        ts = jax.vmap(lambda s: s.ts, in_axes=0)(y_sol).squeeze()
        reduce_turns_loss = turn_loss(ts, 20)
        loss = loss + 0.3 * reduce_turns_loss

    return loss

@eqx.filter_jit
def turn_loss(ts: jax.Array, max_turns: int) -> jax.Array:
    """

    Args:
        ts (jax.Array): The game lengths (shape: (batch_size, ).
        max_turns (int): The maximal game length.

    Returns:
        A loss mapping the average game length to [0, 1].
    """
    min_turns = 1
    return (jnp.mean(ts) - min_turns) / (max_turns - min_turns)


@eqx.filter_jit
def calculate_KL_loss(x: jax.Array, carry: jax.Array) -> jax.Array:
    """
    Helper function that unwraps the distribution parameters from the carry and calculates the KL divergence between
    the predicted parameters and the parameters estimated from the distribution.

    Args:
        x (jax.Array): The dice throws associated with the carry.
        carry (jax.Array): The parameters of the categorical distribution describing the bids.
                Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
                The second and third arrays are padded with zeroes to match the shape of the first array.

    Returns:
        jax.Array: The KL divergence between the predicted and estimated distribution parameters.
    """
    y = get_all_player_y(carry)
    theta_pred = y[..., :-1]  # Remove the bluff parameter
    return jnp.mean(jnp.array([kullback_leibler(x, tp) for tp in theta_pred]))


def get_all_player_y(carry):
    """
    Get the distribution parameters for all players in the game.

    Args:
        carry (jax.Array): The parameters of the categorical distribution describing the bids.
                Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
                The second and third arrays are padded with zeroes to match the shape of the first array.

    Returns:
        jax.Array: The distribution parameters for all players in the game (shape: [n_players, n_faces + 1])
    """
    player_indices = jnp.arange(0, carry.shape[0] - 2, 1)
    y, _, _ = DiscreteDiceGame.unwrap_carry(carry, player_indices)
    return y


@eqx.filter_jit
def calculate_MSE_loss(x: jax.Array, carry: jax.Array) -> jax.Array:
    """
    Helper function that unwraps the distribution parameters from the carry and calculates the MSE between
    the predicted parameters and the parameters estimated from the distribution.

    Args:
        x (jax.Array): The dice throws associated with the carry.
        carry (jax.Array): The parameters of the categorical distribution describing the bids.
                Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
                The second and third arrays are padded with zeroes to match the shape of the first array.

    Returns:
        jax.Array: The KL divergence between the predicted and estimated distribution parameters.
    """
    y = get_all_player_y(carry)
    theta_pred = y[:-1]  # Remove the bluff parameter
    theta_estimated = jit_theta_from_throw(x, theta_pred.shape[0])
    return jnp.mean((theta_pred - theta_estimated)**2)


@eqx.filter_jit
def calculate_bluff_loss(carry: jax.Array) -> jax.Array:
    """
    Helper function that unwraps the distribution parameters from the carry and calculates the loss associated with
    bluffing too much.

    Args:
        carry (jax.Array): The parameters of the categorical distribution describing the bids.
                Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
                The second and third arrays are padded with zeroes to match the shape of the first array.

    Returns:
        jax.Array: The KL divergence between the predicted and estimated distribution parameters.
    """
    _, _, game_state = DiscreteDiceGame.unwrap_carry(carry, 0)
    return jnp.maximum(game_state, jnp.array([0]))


@eqx.filter_jit
def categorical_bluff_loss(carry: jax.Array, epsilon: float = 1e-5) -> Union[float, jax.Array]:
    """
    Calculate the bluff loss based on the categorical cross entropy between the probability of challenging correctly and
    actually challenging correctly.

    Args:
        carry (jax.Array): The parameters of the categorical distribution describing the bids.
                Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
                The second and third arrays are padded with zeroes to match the shape of the first array.
        epsilon (float): A small value to avoid numerical instabilities when calculating the loss.

    Returns:
        float: The categorical cross entropy loss based on the bluff.
    """
    targets = jnp.array([0.0, 1.0])
    game_states = carry[:, 1, 0]
    pseudo_probability = jnp.mean(game_states)
    logits = jnp.log(jnp.array([1-pseudo_probability, pseudo_probability])) + epsilon
    return optax.losses.safe_softmax_cross_entropy(logits, targets)


@eqx.filter_jit
def MSE_bluff_loss(carry: jax.Array) -> Union[float, jax.Array]:
    """
    Calculate the bluff loss based on the MSE between the probability of being challenged incorrectly and
    being challenged incorrectly.

    Args:
        carry (jax.Array): The parameters of the categorical distribution describing the bids.
                Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
                The second and third arrays are padded with zeroes to match the shape of the first array.
    Returns:
        float: The MSE based on the bluff.
    """
    targets = -1  # unsuccessful challenge
    game_states = carry[:, 1, 0]
    return jnp.mean(jnp.abs((targets - game_states)))


@eqx.filter_jit
def unwrap_and_loglikelihood(x: jax.Array, carry: jax.Array) -> jax.Array:
    """
    Helper function that unwraps the distribution parameters from the carry.

    Args:
        x (jax.Array): The dice throws associated with the carry.
        carry (jax.Array): The parameters of the categorical distribution describing the bids.
                Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
                The second and third arrays are padded with zeroes to match the shape of the first array.

    Returns:
        jax.Array: The loglikelihood of throwing the throw `x` given the parameter description in `carry`.
    """
    y = get_all_player_y(carry)
    loss = jit_calculate_loglikelihood(x, y[..., :-1])
    return loss
