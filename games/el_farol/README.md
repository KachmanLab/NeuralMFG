# El Farol bar problem 🍻

The El Farol bar problem is a problem in game theory where a fixed population wants to go to a bar. However, an agent only has a good time at the bar if it's not too crowded. Therefore, the agent loses if more than `threshold`% of the population goes to the bar. 

## ℹ️ Formal setup
The formal setup consists of the _game rules_, the _population setup_ and the _goal of the game_. 

### 🎲 Game rules
The game has three rules (copy-pasted from the definition on [Wikipedia](https://en.wikipedia.org/wiki/El_Farol_Bar_problem)):

1. If **less** than `threshold`% of the population go to the bar, they all have more fun than if they stayed home.
2. If **more** than `threshold`% of the population go to the bar, they all have less fun than if they stayed home.
3. All agents must decide at the same time whether to go or not, with no knowledge of others' choices.

### 📊 Population setup

The population consists of `N` agents, where each agent has an intention probability `\mu` of wanting to go to the bar. Despite their intentions, the agents can be prevented from, or being (peer) pressured into, actually going to the bar. This noise is introduced with scale `\sigma`, such that the actual probability `p_i` of agent `i` going to the bar is drawn from a Gaussian distribution with mean `\mu` and scale `\sigma`.  

As such, the game starts with the initial distribution of intentions `\mu` that are drawn from some Gaussian. Going from these initial values, the agents adapt their intentions to maximize their enjoyment from either going to the bar or staying at home.


### 🎯 The goal of the game
The goal of the game is to learn the distribution of intentions `\bm{\mu}` that minimize the loss of going to a too crowded bar. Using a nODE and an iterated setup, we can train towards modelling the probability of exactly `M` agents visiting the bar, learning the hidden parameters of the true distribution that was used to generate `M`.

## 🐎 Getting started
This game has three entry points. The standard mean-field game can be run via [`bar_example.py`](https://github.com/KachmanLab/MFGSandbox/tree/main/games/el_farol/bar_example.py). The combination of the mean-field mechanics and the data is implemented in [`bar_nODE.py`](https://github.com/KachmanLab/MFGSandbox/tree/main/games/el_farol/bar_nODE.py). You can plot the training data by running [`bar_data.py`](https://github.com/KachmanLab/MFGSandbox/tree/main/games/el_farol/bar_data.py).
