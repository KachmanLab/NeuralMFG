import numpy as np
from games.liars_dice.dice_game_data import generate_data


def test_data_generation():
    N = 1000
    D = 2000
    max_pips = 20
    outcomes = generate_data(N, D, max_pips=max_pips)

    assert outcomes.shape == (N, D)
    assert np.all((1 <= outcomes) & (outcomes <= max_pips))
