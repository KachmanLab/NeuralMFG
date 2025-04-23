import jax
import jax.numpy as jnp
import jax.random as jrandom

from typing import Union


def quantity_from_1h_array(a: jax.Array) -> jax.Array:
    """
    Get the one hot (1h) encoded value from `a`. Typically used to calculate proposed quantity from the bid `a`.

    Args:
        a (jax.Array): Arraylike, typically the bid.

    Returns:
        jax.Array[int]: The quantity of the bids (shape: (1,))
    """
    (idx,) = jnp.nonzero(a, size=1)
    return a[idx].astype(int)


def quantity_from_array(value: Union[int, jax.Array], a: jax.Array) -> Union[jax.Array, int]:
    """
    Calculate the number of occurrences of `value` in array `a`.
    Typically used to count the number of occurrences of a throw outcome (`value`) in all dice_outcomes (`a`).
    Args:
        value (Union[int, jax.Array]): Any value.
        a (jax.Array): Arraylike, typically dice outcomes.

    Returns:
        Union[jax.Array, int]:
    """
    a = a.flatten()
    idx = jnp.where(a == value, jnp.ones_like(a), jnp.zeros_like(a))
    return jnp.sum(idx)


def face_from_array(a: jax.Array) -> jax.Array:
    """
    Calculate proposed face from the bid `a`.

    Args:
        a (jax.Array): Arraylike, typically the bid.

    Returns:
        jax.Array[int]: The face of the bids (shape: (1,))
    """
    (idx,) = jnp.nonzero(a, size=1)
    return (idx + 1).astype(int)


def increment_bid(a: jax.Array, max_count: int, d_face: int = 0, d_count: Union[int, jax.Array] = 0) -> jax.Array:
    """
    Increment the bid while keeping track of its validity.

    Args:
        a (jax.Array): The bid (shape: (n_faces, ))
        max_count (int): The maximum possible count of dice in the game
        d_face (int): The face increment
        d_count (int): The count increment

    Returns:
        jax.Array: The incremented bid.
    """
    (idx,) = jnp.nonzero(a, size=1)
    count = a[idx].astype(int)

    # Bookkeeping the possible entries
    count = jnp.minimum(max_count, count + d_count)
    idx = jnp.minimum(a.shape[0], idx + d_face)

    # Return a new bid
    incremented_bid = jnp.zeros_like(a)
    return incremented_bid.at[idx].set(count)


def estimate_total_count(face: Union[int, jax.Array], p: jax.Array, n: Union[int, jax.Array],
                         dice_outcome: jax.Array, key: jrandom.PRNGKey) -> Union[int, jax.Array]:
    """
    Estimate the total number of occurrences of `face` in all dice thrown.

    Args:
        face (Union[int, jax.Array]): The dice face of interest.
        p (jax.Array): The vector of dice throw probabilities.
        n (Union[int, jax.Array]): The total number of dice thrown.
        dice_outcome (jax.Array): The player's own dice hand.
        key (jrandom.PRNGKey): A PRNG key.

    Returns:
        Union[int, jax.Array]: The estimated quantity based on the player's own hand and the players beliefs about the
            dice odds.
    """
    # Estimate the outcomes of the other players given the agent's parameters (add 1 to cast idx to dice outcomes)
    agent_beliefs = jrandom.categorical(key, jnp.log(p), shape=(n,)) + 1

    # Concatenate the beliefs about the other players' dice with our own hand to maximize the information
    all_outcomes = jnp.concatenate((dice_outcome, agent_beliefs))

    # Return the number of occurrences of `face`
    return quantity_from_array(face, all_outcomes)
