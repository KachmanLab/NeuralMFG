import jax
import jax.numpy as jnp


class MeetingTime:
    """Simulate a mean field game that estimates the time of a meeting."""
    def __init__(self, scheduled_time: float, start_threshold: float, nu: float=1):
        assert 0 < start_threshold <= 1.0, \
            f"The threshold should be a float 0 < threshold <= 1.0, received: {start_threshold}."
        self.scheduled_time = scheduled_time
        self.start_threshold = start_threshold
        self.nu = nu

    def drift(self, t, y, args):
        """The agents' control, moves in the direction of the negative gradient of the loss to minimize the loss."""
        arrival_times = y
        return (jax.vmap(lambda a_t, s_t: -jax.grad(self.loss)(a_t, s_t), (0, None))
                (arrival_times, self.calculate_starting_time(arrival_times)))

    def diffusion(self, t, y, args):
        """The agents' diffusion term (constant)."""
        return jnp.sqrt(2 * self.nu)

    def calculate_starting_time(self, arrival_times: jax.Array):
        """Calculate the starting time of the meeting. The meeting starts if the number of attendees exceeds the
        threshold."""
        return jnp.percentile(arrival_times, self.start_threshold * 100)

    def loss(self, arrival_time: float, starting_time: float, a: float = 1.0, b: float = 1.0, c: float = 1.0) \
            -> jax.Array:
        """Calculate the losses inflicted due to being too late or too early to the meeting."""
        return (a * self.reputation_effect(arrival_time) +
                b * self.personal_inconvenience(arrival_time, starting_time) +
                c * self.waiting_time(arrival_time, starting_time))

    def reputation_effect(self, arrival_time: float):
        """A cost (reputation effect) of lateness in relation to the scheduled time `arrival time`"""
        cost = arrival_time - self.scheduled_time
        return jnp.where(cost >= 0.0, cost, 0.0)

    @staticmethod
    def personal_inconvenience(arrival_time: float, starting_time: float):
        """A cost (personal inconvenience) of lateness in relation to the actual `starting time` of the meeting"""
        cost = arrival_time - starting_time
        return jnp.where(cost >= 0.0, cost, 0.0)

    @staticmethod
    def waiting_time(arrival_time: float, starting_time: float):
        """A waiting time cost that corresponds to the time lost waiting to reach the `starting time`"""
        cost = starting_time - arrival_time
        return jnp.where(cost >= 0.0, cost, 0.0)

