import diffrax
import equinox as eqx

import jax


from games.el_farol.bar_data import cap
from utilities.MultiLayerPerceptron import Func


class NeuralODE(eqx.Module):
    func: Func
    data_size: int
    drift: diffrax.ODETerm

    def __init__(self, data_size, width_size, depth, drift: diffrax.ODETerm, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(data_size, data_size, width_size, depth, key=key, activation=jax.nn.leaky_relu,
                         final_activation=lambda x: x)
        self.drift = drift
        self.data_size = data_size

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.MultiTerm(self.drift, diffrax.ODETerm(self.func)),
            diffrax.Euler(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return cap(solution.ys)
