"""
Implementation of the 1 Euro Filter (Casiez et al., 2012).
A first-order low-pass filter with an adaptive cutoff frequency.
Designed specifically to minimize jitter and lag in interactive systems by
attenuating high-frequency sensor noise at low velocities while
preserving signal integrity during rapid ballistic movements.
"""

import math


def smoothing_factor(t_e, cutoff):
    """Calculates the alpha smoothing factor for the low-pass filter."""
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    """Applies standard exponential smoothing."""
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        Initializes the filter states and parameters.
        :param min_cutoff: Minimum cutoff frequency (Hz) to eliminate resting jitter.
        :param beta: Speed coefficient to reduce lag during rapid movement.
        :param d_cutoff: Cutoff frequency for the derivative filter.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        # State variables
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def __call__(self, t, x):
        """
        Computes the filtered output for the current time step.
        :param t: Current timestamp (seconds).
        :param x: Current noisy sensor reading (mm).
        :return: Filtered position reading.
        """
        t_e = t - self.t_prev
        if t_e <= 0.0:
            return x

        # 1. Filter the derivative (velocity)
        alpha_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(alpha_d, dx, self.dx_prev)

        # 2. Estimate adaptive cutoff frequency based on velocity magnitude
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # 3. Filter the primary signal
        alpha = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(alpha, x, self.x_prev)

        # 4. Update states
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat
