import time
from typing import Union

import numpy as np

from games.el_farol.BarProblem import BarProblem
from games.el_farol.bar_example import bar_boxplots
from games.el_farol.bar_data import generate_probabilities, probabilities_to_attendance_rates
from games.el_farol.neuralODE import NeuralODE

from utilities.plot import plot_loss
from utilities.train import dataloader, make_step

import jax.numpy as jnp
import jax.random as jrandom

import jax
import optax
import equinox as eqx
from diffrax import ODETerm


def main(steps: int = 500, suppress: bool = False):
    # Random state management
    np.random.seed(42)
    key = jrandom.PRNGKey(10)
    train_key, val_key, model_key = jrandom.split(key, 3)

    # Game parameters
    N = 6
    M = 7
    n_observations = 10

    threshold = 0.9  # 0.9

    # Data parameters
    dataset_size = 10
    val_size = 100

    mu = 0.2  # 0.2  # The mean intention of going to the bar
    sigma = 0.01  # Noise around this intention

    k = 2
    theta = 0.01

    # Model parameters
    batch_size = 4
    lr = 5e-4
    print_every = 100

    # Generate initial distribution
    y0 = jnp.array(np.random.random(N * M))

    # Generate target distribution
    ps = jnp.array([generate_probabilities(mu, sigma, k, theta, N, M)[-1] for _ in range(dataset_size)])
    ys = probabilities_to_attendance_rates(ps, n_observations, train_key)

    ps_validation = jnp.array([generate_probabilities(mu, sigma, k, theta, N, M)[-1] for _ in range(val_size)])
    ys_validation = probabilities_to_attendance_rates(ps_validation, n_observations, val_key)

    # Initialize the bar
    t0 = 1
    t1 = 15
    ts = np.linspace(t0, t1, 15)
    el_farol = BarProblem(threshold)
    drift = ODETerm(el_farol.drift)

    # Create neuralODE
    data_key, model_key, loader_key, loss_key = jrandom.split(model_key, 4)
    model = NeuralODE(N * M, 8, 3, drift, key=model_key)

    optim = optax.adabelief(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    # Train loop
    train_loss = np.zeros(steps)
    validation_loss = np.zeros(steps)

    for i, (step, yi) in enumerate(zip(range(steps), dataloader(ys, batch_size, key=loader_key))):
        key, loss_key = jrandom.split(loss_key, 2)

        start = time.time()
        train_loss[i], model, opt_state = make_step(ts, y0, yi, model, optim, opt_state, grad_loss)
        end = time.time()

        validation_loss[i], _ = grad_loss(model, ts, y0, ys_validation)

        if (step % print_every) == 0 or step == steps - 1 and not suppress:
            print(f"Step: {step}, train loss: {train_loss[i]:.4f}, computation time: {(end - start):.4f}"
                  f"\t- validation loss: {validation_loss[i]:.4f}")

    if not suppress:
        plot_loss(train_loss, validation_loss, r'./games/el_farol/figures/bar_nODE_loss.pdf')

        model_ys = model(ts, y0)
        bar_boxplots(model_ys, threshold, r'./games/el_farol/figures/bar_nODE.pdf', data_mean=mu, left=False)


@eqx.filter_value_and_grad
def grad_loss(model: NeuralODE, ti: Union[np.ndarray, jax.Array], y0: jax.Array, yi: jax.Array) -> jax.Array:
    """
    Calculate the loss and gradients of the bar game. Uses `jax`' vmap for efficiency.

    Args:
        model (NeuralODE): The neural ODE model.
        ti (jax.Array): The timesteps.
        y0 (jax.Array): The initial distribution of attendance rates.
        yi (jax.Array): A collection attendance rates (shape: (batch, n_observations)).

    Returns:
        Tuple[jax.Array, jax.Array]: The loss and the gradients of the bar problem.
    """
    y_pred = jax.vmap(model, in_axes=(None, 0))(ti, jnp.repeat(y0[jnp.newaxis, :], yi.shape[0], axis=0))
    estimated_attendance_rates = estimate_attendance_rates(yi, y_pred.shape[-1])
    return jnp.mean((y_pred[:, -1, :] - estimated_attendance_rates) ** 2)


def estimate_attendance_rates(attendance: jax.Array, n: Union[int, jax.Array]) -> jax.Array:
    """
    Estimate the attendance rates from a vector of observed attendances. The total number of agents that can attend is
    equal to `n`.

    Args:
        attendance (jax.Array): A vector of observed attendances.
        n (int): The total number of agents that can attend.

    Returns:
        jax.Array: A vector of estimated attendance rates.
    """
    return jnp.mean(attendance/n, axis=1)[:, np.newaxis]


if __name__ == '__main__':
    main(steps=300)
