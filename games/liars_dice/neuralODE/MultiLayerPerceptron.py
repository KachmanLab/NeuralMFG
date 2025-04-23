import jax
import jax.nn as jnn
import jax.numpy as jnp

import equinox as eqx

from games.liars_dice.DiceGame import DiscreteDiceGame


class Func(eqx.Module):
    """A multilayer perceptron wrapped in facade to communicate with the dice game."""
    mlp: eqx.nn.MLP
    n_players: int

    def __init__(self, in_size, out_size, width_size, depth, *, key,
                 activation=jnn.softplus, final_activation=lambda x: x, n_players: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            final_activation=final_activation,
            key=key,
        )
        self.n_players = n_players

    def __call__(self, t: int, carry: jax.Array, args: dict):
        """
        Apply the neural network, wrapping its output in the carry to prevent a broadcast in the ODE solve.

        Args:
            t (int): The time.
            carry (jax.Array): The carry,
                Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
                The second and third arrays are padded with zeroes to match the shape of the first array.
            args (dict): A dictionary of game arguments.

        Returns:

        """
        # Unwrap the carry
        y, previous_bid, _ = DiscreteDiceGame.unwrap_carry(carry, t)

        dice_roll = DiscreteDiceGame.get_dice_roll(t, self.n_players, args, None)

        input_vector = jnp.concatenate((y, previous_bid, dice_roll))

        network_output = self.mlp(input_vector)

        # Wrap the carry
        return DiscreteDiceGame.wrap_carry(network_output, t, self.n_players)
