from typing import Union

import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

from games.liars_dice.DiceGame import DiceGame
from games.liars_dice.DiceGame.probability import jit_at_least_q
from games.liars_dice.DiceGame.DiceGameMechanics import (quantity_from_1h_array, face_from_array, quantity_from_array,
                                                         estimate_total_count)


class DiscreteDiceGame(DiceGame):
    """A DiceGame variant with discrete turns."""
    def __init__(self, n_players: int, n_dice_per_player: int, max_pips: int, nu: float = 1.0, debug: bool = False):
        super().__init__(n_players, n_dice_per_player, max_pips, nu, debug)

    @staticmethod
    def get_dice_roll(t: Union[int, jax.Array], n_players, args: dict, key):
        player_idx = jnp.array(t, int) % n_players
        dice_roll = args['dice_outcomes'][player_idx]
        return dice_roll

    def make_bid(self, previous_bid: jax.Array, dice_outcome: jax.Array, dice_roll_probabilities: jax.Array, *,
                 key: jrandom.PRNGKey, debug: bool = False) -> jax.Array:
        """Make a bid based on the players' own dice outcomes and the player's belief about the outcome distribution."""
        # Get the info from the previous bid
        previous_face = face_from_array(previous_bid)
        previous_count = quantity_from_1h_array(previous_bid)

        # Calculate the total number of dice in the hands of the other players
        remaining_dice = self.max_quantity - self.n_dice_per_player

        # Estimate the total number of occurrences of face, leveraging the agent's own information
        count_estimate = estimate_total_count(previous_face, dice_roll_probabilities, remaining_dice, dice_outcome, key)

        # The new bid should exceed the previous bid (by one for simplicity, to learn the rest via bluffing)
        proposed_count = previous_count + 1

        # Check if the proposed count is less than or equal to the count estimate
        count_bool = jnp.less_equal(proposed_count, count_estimate)

        # If the proposed count exceeds the estimate, we increase the face. Else, we increase the count
        face = jnp.where(count_bool, previous_face, previous_face + 1,)
        count = jnp.where(count_bool, proposed_count, previous_count,)

        # If we wanted to increase the face, but we couldn't -> we challenge the previous bid!
        count = jnp.where(jnp.greater(face, self.max_pips), -1, count)

        # If the proposed count seems ridiculous -> we challenge the previous bid!
        required_quantity = jnp.maximum(previous_count - quantity_from_array(previous_face, dice_outcome), 1)
        bp = jit_at_least_q(required_quantity, remaining_dice, dice_roll_probabilities[previous_face-1])
        count = jnp.where(jnp.less(bp, 0.3), -1, count)

        # If the proposed count on our new face seems ridiculous -> we challenge the previous bid!
        required_quantity = jnp.maximum(count - quantity_from_array(face, dice_outcome), 1)
        bp = jit_at_least_q(required_quantity, remaining_dice, dice_roll_probabilities[face-1])
        count = jnp.where(jnp.less(bp, 0.3), -1, count)

        if debug and count < 0:  # Debug prints
            print(f"Bid challenged!")
            print(f"\tFace: {previous_face[0]}")
            print(f"\tOwn hand: {np.array(dice_outcome).astype(int)}")
            print(f"\tRequired quantity {required_quantity[0]}, pc {previous_count}, o"
                  f"h {quantity_from_array(previous_face, dice_outcome)}")
            print(f"\tEstimated count: {count_estimate}")
            print(f"\tDice odds: {list(np.array(dice_roll_probabilities))} "
                  f"(sum = {np.round(np.sum(dice_roll_probabilities), 3)})")
            print(f"\tBid probability: {np.array(bp)[0]:.3f}")

        # Submit the bid
        bid = jnp.zeros_like(previous_bid)
        return bid.at[face-1].set(count)


class ContinuousDiceGame(DiceGame):
    """A DiceGame variant with continuous turns."""
    def __init__(self, n_players: int, n_dice_per_player: int, max_pips: int, nu: float = 1.0, debug: bool = False):
        super().__init__(n_players, n_dice_per_player, max_pips, nu, debug)

    def get_dice_roll(self, t: Union[int, jax.Array], args: dict, key):
        dice_roll = jrandom.categorical(key, jnp.log(args['y_true']), shape=(self.n_dice_per_player,))
        return dice_roll

    def make_bid(self, previous_bid: jax.Array, dice_outcome: jax.Array, dice_roll_probabilities: jax.Array, *,
                 key: jrandom.PRNGKey, debug: bool = False) -> jax.Array:
        raise NotImplementedError
