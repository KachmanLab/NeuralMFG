from typing import Tuple, Any

import jax
import jax.random as jrandom

import equinox as eqx

from games.liars_dice.neuralODE import NeuralODE
from games.liars_dice.neuralODE.loss import grad_loss


@eqx.filter_jit
def make_step(carry0: jax.Array, yi: jax.Array, model: NeuralODE, optim: Any, opt_state: Any,
              key: jrandom.PRNGKey) -> Tuple[Any, Any, Any]:
    """
    Calculate and backpropagate the loss over a batch of liar's dice games.

    Args:
        carry0 (jax.Array): The initial carry.
            Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
            The second and third arrays are padded with zeroes to match the shape of the first array.
        yi (jax.Array): A collection of dice throws (shape: (size, n_players, n_dice)).
        model (NeuralODE): The neural ODE model.
        optim (Any): The optimizer.
        opt_state (Any): The initialized optimizer state.
        key (jrandom.PRNGKey): A random key.


    Returns:
        Tuple[Any, Any, Any]: The loss, model and newly updated optimizer state.
    """
    loss, grads = grad_loss(model, carry0, yi, key)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state
