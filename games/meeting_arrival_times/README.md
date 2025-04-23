# Meeting arrival times ⏱️
The repository currently contains code that models the “Meeting arrival times” game, a simple mean-field game. The example is introduced in the book “Mean Field Games and Applications” by Olivier Guéant, Jean-Michel Lasry and Pierre-Louis Lions [1, p. 9]. 

## 🎲 Game rules
A meeting is scheduled for a certain time $t$, but only starts after a certain
percentile of players has arrived. The actual starting time is represented by $T$. Each player $i$ has their own arrival time, $\tilde{\tau}_i = \tau_i + \epsilon_i$, where $\tau_i$ is the planned arrival time of player $i$ and $\epsilon_i$ is the noise associated with that planning. 

## 📊 Population setup
The data are generated in `arrival_times_data.py`. The data represent the outcomes of the games at turn 15 and differs between the games. Having  generative distributions and  players per distribution, there are 42 players in the game. The graphical (left) and generative (right) model of the data generation is shown below. This file can be run separately to generate a figure of the generation process.

![The graphical and generative model of the data generation](https://github.com/KachmanLab/MFGSandbox/tree/main/figures/data_generation.png)

## 🎯 The goal of the game
Given the three times, $t$, $\tau_i$ and $T$, each player $i$ aims to minimize the cost associated with being too early or too late to the meeting.


## 🐎 Getting started
This game has three entry points. The standard mean-field game can be run via [`arrival_times_example.py`](https://github.com/KachmanLab/MFGSandbox/tree/main/games/meeting_arrival_times/arrival_times_example.py). The combination of the mean-field mechanics and the data is implemented in [`arrival_times_nODE.py`](https://github.com/KachmanLab/MFGSandbox/tree/main/games/meeting_arrival_times/arrival_times_nODE.py). You can plot the training data by running [`arrival_times_data.py`](https://github.com/KachmanLab/MFGSandbox/tree/main/games/meeting_arrival_times/arrival_times_data.py). In addition, this game is featured in the [`getting_started.ipynb`](https://github.com/KachmanLab/MFGSandbox/tree/main/games/getting_started.ipynb) notebook.