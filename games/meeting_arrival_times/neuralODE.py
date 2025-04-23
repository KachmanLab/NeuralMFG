from typing import Union

import diffrax
import equinox as eqx

import numpy as np

import jax
import jax.random as jrandom
import jax.numpy as jnp

from utilities.MultiLayerPerceptron import Func


class NeuralODE(eqx.Module):
    func: Func
    drift: diffrax.ODETerm
    diffusion: diffrax.ControlTerm

    def __init__(self, data_size, width_size, depth, drift: diffrax.ODETerm, diffusion: diffrax.ControlTerm,
                 *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(in_size=data_size, out_size=data_size, width_size=width_size, depth=depth, key=key)
        self.drift = drift
        self.diffusion = diffusion

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.MultiTerm(self.drift, self.diffusion, diffrax.ODETerm(self.func)),
            diffrax.Euler(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys


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
    dataset_size = arrays[0].shape[0]
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
