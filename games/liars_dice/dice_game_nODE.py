import time
import numpy as np
from tqdm import tqdm

import wandb

import jax.numpy as jnp
import jax.random as jrandom

import optax
import equinox as eqx
from diffrax import Euler, SaveAt

from games.liars_dice.dice_game_data import generate_nODE_data, generate_opening_bid
from games.liars_dice.neuralODE import Func, NeuralODE, grad_loss, make_step
from games.liars_dice.DiceGame import analyse_nODE_solution, DiscreteDiceGame

from utilities.plot import plot_loss
from utilities.train import dataloader

import jax
jax.config.update("jax_enable_x64", True)


def main(steps: int = 500, suppress: bool = False, weights_and_biases: bool = False, **kwargs):
    # Random state management
    key = jrandom.PRNGKey(42) if 'key' not in kwargs else kwargs['key']
    seed = 10 if 'seed' not in kwargs else kwargs['seed']
    np.random.seed(seed)

    data_key, model_key, loader_key = jrandom.split(key, 3)

    # Game parameters, optional via kwargs
    n_players = 5 if 'n_players' not in kwargs else kwargs['n_players']
    n_dice = 30 if 'n_dice' not in kwargs else kwargs['n_dice']
    n_faces = 6 if 'n_faces' not in kwargs else kwargs['n_faces']

    # Learning parameters
    batch_size = 32 if 'batch_size' not in kwargs else kwargs['batch_size']
    lr = 5e-4 if 'lr' not in kwargs else kwargs['lr']

    if 'scheduler' in kwargs:  # Currently only supports 'cosine_onecycle_schedule'
        scheduler = getattr(optax.schedules, kwargs['scheduler'])(transition_steps=steps, peak_value=lr)
    else:
        scheduler = lr

    optim = getattr(optax, "adabelief" if "optimizer" not in kwargs else kwargs["optimizer"])(scheduler)
    dataset_size = 1000 if 'dataset_size' not in kwargs else kwargs['dataset_size']
    reduce_turns = False if 'reduce_turns' not in kwargs else kwargs['reduce_turns']
    val_size = 100
    print_every = 100

    # Generate data
    # 1: Target distribution
    y_true = np.array([0.1, 0.2, 0.0, 0.4, 0.1, 0.2, ]) if 'y_true' not in kwargs else kwargs['y_true']

    # 2: Labels
    ys = generate_nODE_data(dataset_size, n_players, n_dice, y_true)
    ys_val = generate_nODE_data(val_size, n_players, n_dice, y_true)

    # 3: Initial carry
    lam = 0.1 if 'lam' not in kwargs else kwargs['lam']
    carry0 = generate_opening_bid(n_faces, n_players, lam=lam)

    # Create neuralODE
    in_size = 2 * n_faces + 1 + n_dice  # The face parameters, own_dice roll, the previous bid and the bluff parameter
    out_size = n_faces + 1  # The face parameters and the bluff parameter
    width = 8 if 'width' not in kwargs else kwargs['width']
    depth = 3 if 'depth' not in kwargs else kwargs['depth']
    func = Func(in_size=in_size, out_size=out_size, width_size=width, depth=depth, key=model_key, n_players=n_players)

    game = DiscreteDiceGame(n_players, n_dice, n_faces, debug=False)
    model = NeuralODE(func, game.drift, Euler(), reduced_turns=reduce_turns)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    # Train loop
    train_loss = np.zeros(steps)
    validation_loss = np.zeros(steps)

    for i, (step, yi) in tqdm(enumerate(zip(range(steps), dataloader(ys, batch_size, key=loader_key))),
                              total=steps, desc=f"{model} training silently with {dataset_size} games",
                              disable=not suppress, leave=False):
        # Key management
        key, train_key, val_key = jrandom.split(key, 3)

        start = time.time()
        train_loss[i], model, opt_state = make_step(carry0, yi, model, optim, opt_state, train_key)
        end = time.time()

        validation_loss[i], _ = grad_loss(model, carry0, ys_val, val_key)

        if not weights_and_biases and ((step % print_every) == 0 or step == steps - 1) and not suppress:
            print(f"Step: {step}, train loss: {train_loss[i]:.4f}, computation time: {(end - start):.4f}"
                  f"\t- validation loss: {validation_loss[i]:.4f}")

        if weights_and_biases:
            wandb.log({'train_loss': train_loss[i], 'validation_loss': validation_loss[i]})

    if not suppress:
        plot_loss(train_loss, validation_loss, r'./games/liars_dice/figures/liars_dice_loss.pdf')
        saveat = SaveAt(ts=jnp.arange(0, 80, 1))
        sol = model(carry0, ys_val[0], saveat, key)

        analyse_nODE_solution(sol.ys, n_players)

    return model


if __name__ == '__main__':
    main(batch_size=64, reduce_turns=False)
