from abc import abstractmethod
from functools import partial
from typing import Union, Tuple

import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit

from games.liars_dice.DiceGame.DiceGameMechanics import (quantity_from_1h_array, face_from_array, increment_bid,
                                                         quantity_from_array)
from games.liars_dice.DiceGame.probability import kullback_leibler


class DiceGame:
    """Simulate a game of liar's dice."""

    def __init__(self, n_players: int, n_dice_per_player: int, max_pips: int, nu: float = 1.0, debug: bool = False):
        self.max_pips = max_pips
        self.n_players = n_players
        self.n_dice_per_player = n_dice_per_player
        self.max_quantity = n_players * n_dice_per_player
        self.nu = nu
        self.debug = debug

    def drift(self, t, carry, args):
        """The agents' control, aiming to find the best tradeoff between bluffing and minimizing the risk of being
        challenged.

        Args:
            t (int): the turn index, indicating what player is considered.
            carry (jax.Array): The parameters of the categorical distribution describing the bids.
                Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
                The second and third arrays are padded with zeroes to match the shape of the first array.
            args (dict): Additional arguments for the differential process.

        Returns:
            jax.Array: The drift of the categorical distribution parameters.
        """
        # Generate a random key for making probability estimates
        key = jrandom.PRNGKey(np.random.randint(0, 100, dtype=int))

        # Split the carry in the distribution parameters and the previous bid
        player_index = t % self.n_players
        y, previous_bid, _ = self.unwrap_carry(carry, player_index)

        # Get the dice roll of player `t`
        key, roll_key = jrandom.split(key, 2)
        dice_roll = self.get_dice_roll(t, self.n_players, args, key)

        # Update the game state
        delta_bid, delta_game_state = self.state_update(t, y, dice_roll, previous_bid, args, key)

        argnums = list(range(5, 5 + y.shape[0]))
        loss, grad = jax.value_and_grad(self.turn, argnums=argnums)(t,
                                                                    dice_roll,
                                                                    previous_bid,
                                                                    args['dice_outcomes'],
                                                                    key,
                                                                    *tuple(y))
        negative_gradient = -1 * jnp.array(grad)

        wrapped_gradient = jnp.zeros((self.n_players, y.shape[0]))
        wrapped_gradient = wrapped_gradient.at[jnp.array(player_index, int)].set(negative_gradient)

        # if self.debug:
        #       print(f"Turn {t}\n\tObtained loss: {loss}\n\tobtained gradient: {jnp.array(grad)}")

        return jnp.vstack((delta_bid, delta_game_state, wrapped_gradient))

    def state_update(self, t: Union[int, jax.Array], y: jax.Array, dice_roll: jax.Array, previous_bid: jax.Array,
                     args: dict, key: jrandom.PRNGKey) -> Tuple[jax.Array, jax.Array]:
        """
        Helper function to update the game state.

        Args:
            t (Union[int, float]): The player index. Int in discrete games, float in continuous games.
            y (jax.Array): A flexible number of parameter describing the dice odds.
            dice_roll (jax.Array): The dice outcomes of the current player.
            previous_bid (jax.Array): The bid made by the previous player, in array form.
            args (dict): Additional arguments for the differential process.
            key (jrandom.PRNGKey): A PRNG key.

        Returns:
            Tuple[jax.Array, jax.Array]: The state update.
        """
        # Update the bidding state
        current_bid = self.bid_and_bluff(previous_bid, tuple(y), dice_roll, key, debug=self.debug)
        if self.debug:
            print(f"\tPlayer {t % self.n_players + 1} rolled {dice_roll} and bids {current_bid}")

        # Update the game state, in case a bid has been challenged
        delta_game_state_single = self.check_challenge(current_bid, previous_bid, args['dice_outcomes'])
        delta_game_state_padding = jnp.zeros_like(y)
        delta_game_state_padding += delta_game_state_single

        # Calculate the "derivative" of the bid, practically replacing the previous bid with the current bid
        delta_bid = self.bid_derivative(previous_bid, current_bid, padding=1)  # Padding for bluff parameter
        return delta_bid, delta_game_state_padding

    def diffusion(self, t, y, args):
        """The agents' diffusion term (constant)."""
        return jnp.sqrt(2 * self.nu)

    def turn(self, turn_idx: jax.Array, dice_roll: jax.Array, previous_bid: jax.Array,
             dice_outcomes: jax.Array, key: jrandom.PRNGKey, *y: jax.Array) -> Union[float, jax.Array]:
        """
        Simulate a turn of Liar's dice!

        Args:
            turn_idx (jax.Array): The index of the turn. This is an integer when a discrete game is played.
            dice_roll (jax.Array): The dice outcomes of the current player.
            previous_bid (jax.Array): The bid made by the previous player, in array form.
            dice_outcomes (jax.Array): The outcomes of the dice of all players.
            key (jrandom.PRNGKey): A PRNG key.
            *y (jax.Array): A flexible number of parameter describing the dice odds and the bluff probability.
                Contains: [*n_faces die odds, bluff_probability]

        Returns:
            float: The loss associated with the dice odds and bluff probability.
        """

        current_bid = self.bid_and_bluff(previous_bid, y, dice_roll, key, debug=False)

        # Calculate the loss based on the current bid
        return self.loss(current_bid, dice_outcomes, previous_bid, jnp.array(y[:self.max_pips]))

    def bid_and_bluff(self, previous_bid: jax.Array, y: Tuple[jax.Array], dice_roll: jax.Array, key: jrandom.PRNGKey,
                      debug: bool) -> jax.Array:
        """
        Place a bid and add bluff.

        Args:
            previous_bid (jax.Array): The previous bid (shape: (n_faces, ))
            y (Tuple[jax.Array]): The distribution parameters (shape: (n_faces + 1, )).
            dice_roll (jax.Array): The dice outcomes of the current player.
            key (jrandom.PRNGKey): A PRNG key.
            debug (bool): True if debug statements should be printed.

        Returns:
            jax.Array: The proposed bid.
        """
        # Unpack parameters
        dice_roll_parameters = jnp.array(y[:-1])  # The probabilities associated with each dice outcome
        bluff_lamda = y[-1]  # The parameter for the bluffing distribution

        # Key management
        bid_key, bluff_key = jrandom.split(key, 2)

        # Make a bid based on the agent's belief about the distribution of dice and their own throw.
        current_bid = self.make_bid(previous_bid, dice_roll, dice_roll_parameters, key=bid_key, debug=debug)

        # Bluff if the previous bid is not challenged
        d_count = jnp.where(jnp.sum(current_bid) <= 0, 0, self.bluff(bluff_lamda, key=bluff_key))

        # Add bluff to the bid
        return increment_bid(current_bid, max_count=self.max_quantity, d_count=d_count)

    def loss(self, current_bid: jax.Array, dice_outcomes: jax.Array, previous_bid: jax.Array, y_estimated: jax.Array,
             a: float = 1.0, b: float = 1.0, c: float = 1.0, d: float = 0.1) -> jax.Array:
        """Calculate the losses from making inaccurate bids."""
        return (a * self.safety_penalty(previous_bid, current_bid) +
                b * self.bluffing_risk(current_bid, dice_outcomes) +
                c * self.niceness_penalty(current_bid) +
                d * kullback_leibler(x=dice_outcomes, theta_pred=y_estimated))[0]

    def safety_penalty(self, previous_bid: jax.Array, current_bid: jax.Array):
        """A penalty for playing on the safe side, making it easier for the next player to make an agreeable bid."""
        cost = self.max_quantity - (quantity_from_1h_array(current_bid) - quantity_from_1h_array(previous_bid))
        return jnp.where(cost >= 0.0, cost, 0.0)

    @staticmethod
    def bluffing_risk(current_bid: jax.Array, dice_outcomes: jax.Array):
        """A penalty for bluffing too much. Bluffing increases the probability of being challenged."""
        quantity = quantity_from_1h_array(current_bid)
        face = face_from_array(current_bid)
        cost = quantity - quantity_from_array(face, dice_outcomes)
        return jnp.where(cost >= 0.0, cost, 0.0)

    def niceness_penalty(self, current_bid: jax.Array):
        """You're too nice! A penalty for not raising the bid towards the maximum number of pips."""
        cost = self.max_pips - face_from_array(current_bid)
        return jnp.where(cost >= 0.0, cost, 0.0)

    @staticmethod
    @abstractmethod
    def get_dice_roll(t: Union[int, jax.Array],  n_players: int, args: dict, key: jrandom.PRNGKey,):
        """
        Get the dice roll of player t.

        Args:
            t (Union[int, float]): The player index. Int in discrete games, float in continuous games.
            n_players (int): The number of players in the game.
            args (dict): A dictionary of game arguments.
            key (jrandom.PRNGKey): A PRNG key.

        Returns:
            jax.Array: The dice outcomes of player t.
        """
        pass

    @abstractmethod
    def make_bid(self, previous_bid: jax.Array, dice_outcome: jax.Array, dice_roll_probabilities: jax.Array, *,
                 key: jrandom.PRNGKey, debug: bool = False) -> jax.Array:
        """Make a bid based on the players' own dice outcomes and the player's belief about the outcome distribution."""
        pass

    @staticmethod
    def bluff(lam: Union[float, jax.Array], *, key: jrandom.PRNGKey) -> Union[jax.Array, int]:
        """Draw the bluff from a Poisson distribution."""
        return jrandom.poisson(key, lam, shape=())

    @staticmethod
    def bid_derivative(previous_bid: jax.Array, current_bid: jax.Array, padding: int = 0) -> jax.Array:
        """Calculate the `derivative` of the bid. Helper function for ODE solve."""
        return jnp.append(current_bid - previous_bid, jnp.zeros(padding))

    @staticmethod
    def check_challenge(current_bid: jax.Array, previous_bid: jax.Array, dice_outcomes: jax.Array) \
            -> Union[int, jax.Array]:
        """
        Verify whether the previous bid is challenged and whether the challenge is correct. Makes a suggestion for the
        game state update. (0 for no challenge, 1 for a successful challenge and -1 for an unsuccessful challenge.

        Args:
            current_bid (jax.Array): The current bid (shape: (n_faces, ))
            previous_bid (jax.Array): The previous bid (shape: (n_faces, ))
            dice_outcomes (jax.Array): The outcomes of the dice of all players.


        Returns:
            int: A game state update.
        """
        face = face_from_array(previous_bid)
        q_challenged = quantity_from_1h_array(previous_bid)
        q_total = quantity_from_array(face, dice_outcomes)  # multiply

        q_change = jnp.where(jnp.sum(current_bid) <= 0, 1, 0)

        return jnp.where(q_challenged <= q_total, q_change * -1, q_change)

    @staticmethod
    def unwrap_carry(carry: jax.Array, t: Union[int, jax.Array]) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        A helper function to unwrap the carry.

        Args:
            carry (jax.Array): The parameters of the categorical distribution describing the bids.
                Contains: [[previous_bid, 0], [game_state, ..., 0], n_players * s[*n_faces die odds, bluff_probability]].
                The second and third arrays are padded with zeroes to match the shape of the first array.
            t (Union[int, jax.Array]): The turn index/indices.

        Returns:
            Tuple[jax.Array, jax.Array]: the bidding parameters and the previous bid.
        """
        previous_bid = carry[0, :-1]
        game_state = carry[1, 0]

        idx = jnp.array(t + 2, int)
        y = carry[idx]
        y = y.at[..., :-1].set(jax.nn.softmax(y[..., :-1]))  # Dice odds
        y = y.at[..., -1].set(jnp.maximum(y[..., -1], 0.0) )  # Bluff parameter

        return y, previous_bid, game_state

    @staticmethod
    def wrap_carry(distribution_params: jax.Array, t: int, n_players: int) -> jax.Array:
        """
        A helper function to wrap distribution parameters into a carry.

        Args:
            distribution_params (jax.Array): The parameters of the categorical distribution describing the bids.
                Contains: [[*n_faces die odds, bluff_probability].
            t (int): The turn index.
            n_players (int): The number of players to consider.

        Returns:
            The parameters of the categorical distribution describing the bids, padded with zeros for the game state and
            previous bid to prevent a broadcast during the equation solve.
                Contains: [[0,..., 0], [0, ..., 0], n_players * [*n_faces die odds, bluff_probability]].
                The second and third arrays are padded with zeroes to match the shape of the first array.
        """
        wrapped_params = jnp.zeros((n_players, distribution_params.shape[0]))
        wrapped_params = wrapped_params.at[jnp.array(t % n_players, int)].set(distribution_params)
        return jnp.vstack((jnp.zeros((2, distribution_params.shape[0])), wrapped_params))
