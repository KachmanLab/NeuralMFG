from typing import Callable, Union

import diffrax
import equinox as eqx

import jax
import jax.numpy as jnp

from diffrax import AbstractSolver, Event, diffeqsolve, Solution

from games.liars_dice.neuralODE.MultiLayerPerceptron import Func


def challenge_event(t: jax.Array, carry: jax.Array, args: dict, **kwargs):
    """The event function. The event is raised if the function evaluates to `True`."""
    game_state = carry[1, :]
    return jnp.sum(game_state) != 0


class MFGModel(eqx.Module):
    t0: int
    t1: int
    dt: int
    solver: AbstractSolver
    event: Event

    def __init__(self, t0: int, t1: int, dt: int, solver: diffrax.AbstractSolver, **kwargs):
        super().__init__(**kwargs)
        self.t0 = t0
        self.t1 = t1
        self.dt = dt
        self.solver = solver
        self.event = Event(challenge_event)  # Challenging mechanism
        # -> introduces infinity after the mechanism has kicked in

    def __call__(self, carry0: jax.Array,  dice_outcomes: jax.Array, save_at: diffrax.SaveAt, key: jax.random.PRNGKey,)\
            -> Union[jax.Array, Solution]:
        pass


class NeuralODE(MFGModel):
    """A neural ODE for the liar's dice game."""
    term: diffrax.MultiTerm
    reduced_turns: bool

    def __init__(self, func: Func, ode_term: Callable, solver: AbstractSolver, t0: int = 0, t1: int = 1000, dt: int = 1,
                 reduced_turns: bool = False, **kwargs):
        super().__init__(t0, t1, dt, solver, **kwargs)
        self.term = diffrax.MultiTerm(diffrax.ODETerm(ode_term), diffrax.ODETerm(func))
        self.reduced_turns = reduced_turns

    def __call__(self, carry0: jax.Array,  dice_outcomes: jax.Array, save_at: diffrax.SaveAt, key: jax.random.PRNGKey,)\
            -> Solution:
        """
        Apply the neural ODE, solving a single game of Liar's dice.

        Args:
            carry0 (jax.Array): The initial carry.
                Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
                The second and third arrays are padded with zeroes to match the shape of the first array.
            dice_outcomes (jax.Array): A collection of dice throws (shape: (n_players, n_dice)).
            save_at (diffrax.SaveAt): What times to save the solution of the differential equation.
            key (jrandom.PRNGKey): A random key.

        Returns:
            jax.Array: The carry throughout the game of Liar's dice.
        """
        sol = diffeqsolve(terms=self.term,
                          solver=self.solver,
                          t0=self.t0,
                          t1=self.t1,
                          dt0=self.dt,
                          y0=carry0,
                          saveat=save_at,
                          args={'key': key, 'dice_outcomes': dice_outcomes},
                          event=self.event)
        return sol

    def __str__(self):
        name = "neural ODE MFG"
        if self.reduced_turns:
            name += " - reduced turns"
        return name


class MeanFieldGame(MFGModel):
    """A facade pattern for a single mean-field game"""
    term: diffrax.ODETerm

    def __init__(self, ode_term: Callable, solver: AbstractSolver, t0: int = 0, t1: int = 1000, dt: int = 1,
                 **kwargs):
        super().__init__(t0, t1, dt, solver, **kwargs)
        self.term = diffrax.ODETerm(ode_term)

    def __call__(self, carry0: jax.Array,  dice_outcomes: jax.Array, save_at: diffrax.SaveAt, key: jax.random.PRNGKey,):
        """
        Apply the MFG, solving a single game of Liar's dice.

        Args:
            carry0 (jax.Array): The initial carry.
                Contains: [[*n_faces die odds, bluff_probability], [previous_bid, 0], [game_state, ..., 0]].
                The second and third arrays are padded with zeroes to match the shape of the first array.
            dice_outcomes (jax.Array): A collection of dice throws (shape: (n_players, n_dice)).
            save_at (diffrax.SaveAt): What times to save the solution of the differential equation.
            key (jrandom.PRNGKey): A random key.

        Returns:
            jax.Array: The carry throughout the game of Liar's dice.
        """
        sol = diffeqsolve(terms=self.term,
                          solver=self.solver,
                          t0=self.t0,
                          t1=self.t1,
                          dt0=self.dt,
                          y0=carry0,
                          saveat=save_at,
                          args={'key': key, 'dice_outcomes': dice_outcomes},
                          event=self.event)
        return sol.ys

    def __str__(self):
        return "Standard MFG theory"

