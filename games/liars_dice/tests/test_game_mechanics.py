import jax.numpy as jnp
from games.liars_dice.DiceGame.DiceGame import DiceGame


def test_player_loss():
    max_quantity = 6
    dg = DiceGame(1, 6, max_quantity)
    bid = jnp.array([0, 0, 0, 0, 0, 2])
    outcomes = jnp.array([[1, 1, 6], [6, 6, 2]])

    assert dg.bluffing_risk(bid, outcomes) == 0
    assert dg.safety_penalty(jnp.array([0, 0, 0, 0, 2, 0]), bid) == max_quantity
    assert dg.safety_penalty(jnp.array([0, 0, 0, 0, 1, 0]), bid) == max_quantity - 1
    assert dg.niceness_penalty(bid) == 0
    assert dg.niceness_penalty(jnp.array([0, 0, 6, 0, 0, 2])) == 3
