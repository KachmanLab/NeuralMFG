import time
from typing import Union

import numpy as np

from games.meeting_arrival_times.MeetingTime import MeetingTime
from games.meeting_arrival_times.arrival_times_data import generate_arrival_times
from games.meeting_arrival_times.arrival_times_example import game_boxplots
from games.meeting_arrival_times.neuralODE import NeuralODE
from utilities.plot import plot_loss

from utilities.train import make_step

import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import equinox as eqx
from diffrax import ControlTerm, ODETerm, VirtualBrownianTree


def main(steps: int = 500, suppress: bool = False):
    np.random.seed(10)

    mu_init = 12
    mu_target = 15
    sigma = 0.5

    N = 6
    M = 7
    dataset_size = 10
    val_size = 1000

    threshold = 0.8

    batch_size = 4
    lr = 5e-4
    print_every = 100

    # Generate data
    ys = np.array([generate_arrival_times(mu_target, sigma, 2, 0.2, N, M)[-1] for _ in range(dataset_size)])
    ys_validation = np.array([generate_arrival_times(mu_target, sigma, 2, 0.2, N, M)[-1]
                              for _ in range(val_size)])

    # Generate initial distribution
    y0 = np.random.normal(mu_init, sigma, size=N * M)

    # Initialize meeting
    t0 = 1
    t1 = 15
    ts = np.linspace(1, 15, 15)
    meeting = MeetingTime(mu_target, threshold)
    brownian_motion = VirtualBrownianTree(t0, t1, tol=1e-3, shape=(), key=jrandom.PRNGKey(0))
    drift = ODETerm(meeting.drift)
    diffusion = ControlTerm(meeting.diffusion, brownian_motion)

    # Create neuralODE
    key = jrandom.PRNGKey(42)
    data_key, model_key, loader_key = jrandom.split(key, 3)
    model = NeuralODE(N * M, 8, 3, drift, diffusion, key=model_key)

    optim = optax.adabelief(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    # Train loop
    train_loss = np.zeros(steps)
    validation_loss = np.zeros(steps)

    for i, (step, yi) in enumerate(zip(range(steps), dataloader(ys, batch_size, key=loader_key))):
        y0 = np.random.normal(mu_init, sigma, size=N * M)  # Draw a new initial distribution

        start = time.time()
        train_loss[i], model, opt_state = make_step(ts, y0, yi, model, optim, opt_state, grad_loss)
        end = time.time()

        validation_loss[i], _ = grad_loss(model, ts, y0, ys_validation)

        if (step % print_every) == 0 or step == steps - 1 and not suppress:
            print(f"Step: {step}, train loss: {train_loss[i]:.4f}, computation time: {(end - start):.4f}"
                  f"\t- validation loss: {validation_loss[i]:.4f}")

    if not suppress:
        plot_loss(train_loss, validation_loss, r'./games/meeting_arrival_times/figures/arrival_times_loss.pdf')

        model_ys = model(ts, y0)
        game_boxplots(model_ys, r'./games/meeting_arrival_times/figures/arrival_times_nODE.pdf',
                      data_mean=mu_target, left=False)

    return model


@eqx.filter_value_and_grad
def grad_loss(model: NeuralODE, ti: Union[np.ndarray, jax.Array], y0: Union[np.ndarray, jax.Array],
              yi: Union[np.ndarray, jax.Array]):
    """
        Calculate the loss and gradients of the arrival times problem. Uses `jax`' vmap for efficiency.

        Args:
            model (NeuralODE): The neural ODE model.
            ti (jax.Array): The timesteps.
            y0 (jax.Array): The initial distribution of attendance rates.
            yi (jax.Array): A collection attendance rates (shape: (batch, n_observations)).

        Returns:
            Tuple[jax.Array, jax.Array]: The loss and the gradients of the bar problem.
    """
    y_pred = jax.vmap(model, in_axes=(None, 0))(ti, jnp.repeat(y0[jnp.newaxis, :], yi.shape[0], axis=0))
    return jnp.mean((yi - y_pred[:, -1, :]) ** 2)  # Take the last turn



def dataloader(arrays: Union[np.ndarray, jax.Array], batch_size: int, *, key: jrandom.PRNGKey):
    """
    Load the data in randomized batches of `batch_size`.
    Args:
        arrays (np.ndarray): The data.
        batch_size (int): The batch size.
        key (jrandom.PRNGKey): A random key.

    Returns:
        np.ndarray: batched data.
    """
    dataset_size = arrays.shape[0]
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield arrays[batch_perm]
            start = end
            end = start + batch_size


if __name__ == '__main__':
    main(steps=300)
