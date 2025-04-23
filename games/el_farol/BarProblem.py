from typing import Union

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jrandom

from games.el_farol.bar_data import cap


class BarProblem:
    """Simulate a mean field game that estimates the El Farol bar problem."""
    def __init__(self, crowding_threshold: float, nu: float = 1):
        assert 0 < crowding_threshold <= 1.0, \
            f"The threshold should be a float 0 < threshold <= 1.0, received: {crowding_threshold}."
        self.crowding_threshold = crowding_threshold
        self.nu = nu

    def drift(self, t, y, args):
        """The agents' control, moves in the direction of the negative gradient of the loss to minimize the loss."""
        probabilities = y

        # Cap the probabilities
        cap(probabilities)

        # Generate a random key for making attendance estimates
        key = jrandom.PRNGKey(np.random.randint(0, 100, dtype=int))
        attendees = jrandom.bernoulli(key, probabilities)

        # Calculate the percentage of the population that goes to the bar
        percentage = self.calculate_percentage_attendees(attendees)

        # Move into the direction of the negative gradient
        return (jax.vmap(lambda _prob, _perc, _a: -jax.grad(self.loss,)(_prob, _perc, _a, 0.33, 0.33, 0.33),
                         (0, None, 0))(probabilities, percentage, attendees))

    def diffusion(self, t, y, args):
        """The agents' diffusion term (constant)."""
        return jnp.sqrt(2 * self.nu)

    def loss(self, probability: jax.Array, percentage: jax.Array, attendance: jax.Array, a: float = 1.0,
             b: float = 1.0, c: float = 1.0) -> jax.Array:
        """Calculate the losses inflicted due to being at a too crowded bar or missing out on a great evening."""
        return (a * self.crowding_penalty(probability, percentage, attendance) +
                b * self.missed_a_good_evening(probability, percentage, attendance) +
                c * self.peer_pressure(probability, percentage))

    def missed_a_good_evening(self, probability: jax.Array, percentage: jax.Array, attendance: jax.Array)\
            -> Union[float, jax.Array]:
        """Calculate the losses inflicted due to the other agents having a great evening, while you stayed at home."""
        cost = self.crowding_threshold - probability
        cost = jnp.where(cost >= 0.0, cost, 0.0)  # Ensure positivity

        # Only penalize when the bar is not too crowded
        cost = jnp.where(percentage <= self.crowding_threshold, cost, 0.0)

        # Only inflict the loss on the agents that did not go to the bar (attendance = 0)
        return jnp.where(attendance, 0.0, cost)

    def crowding_penalty(self, probability: jax.Array, percentage: jax.Array, attendance: jax.Array)\
            -> Union[float, jax.Array]:
        """Calculate the losses inflicted due to going to a too crowded bar."""
        cost = probability - self.crowding_threshold
        cost = jnp.where(cost >= 0.0, cost, 0.0)  # Ensure positivity

        # Only penalize when the crowding threshold is breached
        cost = jnp.where(percentage > self.crowding_threshold, cost, 0.0)

        # Only inflict the loss on the agents that went to the bar (attendance = 1)
        return jnp.where(attendance, cost, 0.0)

    @staticmethod
    def peer_pressure(probability: jax.Array, percentage: jax.Array,):
        """The loss inflicted due to not conforming with their peers, regardless of going to the bar."""
        return jnp.pow(probability - percentage, 2)

    @staticmethod
    def calculate_percentage_attendees(attendees: jax.Array) -> Union[float, jax.Array]:
        """
        Calculate the percentage of the total population of agents going to the bar.
        Args:
            attendees (jax.Array): A binary vector, indicating whether an agent is going (1) or not (0) to the bar.
                (shape: (n_agents,)).

        Returns:
            float: the percentage of agents that go to the bar.
        """
        return jnp.sum(attendees) / attendees.shape[0]
