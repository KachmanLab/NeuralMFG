# Liar's dice 🎲

In a game of Liar's dice, `N` players throw `M` dice each, and subsequently make a bid about the number of occurrences of a specific die face. The players are merely allowed to increment the number of occurrences, or the number of pips shown on the die face. Each player has information about their own `M` dice, while the faces of the other `(N-1) * M` dice is unknown. The players are allowed to bluff, increasing the bid even if it seems unlikely. However, bluffing increases the risk of being challenged by the next player. Losing a challenge results in the loss of a die, reducing the information available to the player. The player that has the last remaining dice, wins the game.

## 🎲 Game rules

* In the first turn, all players throw their dice and the first player makes a bid.
* Subsequent turns rotate among the players, where each player can decide to either
  1. Challenge the previous bid.
  2. Make a higher bid, either increasing the number of occurrences or the face.

## 📊 Population setup
The dice outcomes are modelled with a categorical distribution. During the game, the players aim to update their beliefs about the parameters of this distribution. Furthermore, the players aim to optimize their bluffing strategy, which follows a Poisson distribution. 

## 🎯 The goal of the game
The players try to prevent losing a die from being challenged, as the player that has the most dice left, wins the game.

## 🐎 Getting started
This game has three entry points. The standard mean-field game can be run via [`dice_game_example.py`](https://github.com/KachmanLab/MFGSandbox/tree/main/games/liars_dice/dice_game_example.py). The combination of the mean-field mechanics and the data is implemented in [`dice_game_nODE.py`](https://github.com/KachmanLab/MFGSandbox/tree/main/games/liars_dice/dice_game_nODE.py). You can plot the training data by running [`dice_game_data.py`](https://github.com/KachmanLab/MFGSandbox/tree/main/games/liars_dice/dice_game_data.py).


