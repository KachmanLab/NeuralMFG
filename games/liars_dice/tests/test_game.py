from games.liars_dice import dice_game_nODE, dice_game_example


def test_nODE():
    dice_game_nODE.main(1, True)


def test_example():
    dice_game_example.main(True)
