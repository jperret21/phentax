# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT

# Credits for the original implementations: Cecilio García Quirós

"""
Waveform conditioning utilities for FFT preparation.
"""

import jax.numpy as jnp
from jax import jit
from jaxtyping import Array


@jit
def planck_taper(N: int | Array, n_left: int | Array, n_right: int | Array) -> Array:
    """
    Generates a Planck window of length N.

    Parameters
    ----------
        N: Total length of the window.
        n_left: Number of points for the left taper (rise).
        n_right: Number of points for the right taper (decay).

    Returns
    -------
        window: Array of shape (N,) with values between 0 and 1.
    """
    x = jnp.arange(N, dtype=jnp.float32)

    # Left taper (rise from 0 to n_left)
    # z = (n_left) * (1/k + 1/(k-n_left))
    # We want transition from 0 to 1.
    # Let's use a safe implementation avoiding division by zero

    def _sigmoid(k, n):
        # k goes from 0 to n
        # Avoid endpoints
        k_safe = jnp.clip(k, 1e-10, n - 1e-10)
        z = n * (1.0 / k_safe + 1.0 / (k_safe - n))
        return 1.0 / (1.0 + jnp.exp(z))

    w = jnp.ones(N)

    # Apply left taper
    mask_left = x < n_left
    w = jnp.where(mask_left, _sigmoid(x, n_left), w)
    w = jnp.where(x == 0, 0.0, w)  # Enforce 0 at start

    # Apply right taper
    # x goes from N-n_right to N
    # map to k going from n_right to 0
    mask_right = x >= (N - n_right)
    k_right = (N - 1) - x
    w = jnp.where(mask_right, _sigmoid(k_right, n_right), w)
    w = jnp.where(x == N - 1, 0.0, w)  # Enforce 0 at end

    return w


@jit
def planck_taper_masked(
    N: int | Array, n_valid: int | Array, n_left: int | Array, n_right: int | Array
) -> Array:
    """
    Generates a Planck window of length N, tapering at the start and at n_valid.

    Parameters
    ----------
        N: Total length of the array.
        n_valid: Index of the last valid point (or number of valid points).
                 The taper will end at index n_valid-1.
        n_left: Number of points for the left taper.
        n_right: Number of points for the right taper.

    Returns
    -------
        window: Array of shape (N,) with values between 0 and 1.
    """
    x = jnp.arange(N, dtype=jnp.float32)

    def _sigmoid(k, n):
        k_safe = jnp.clip(k, 1e-10, n - 1e-10)
        z = n * (1.0 / k_safe + 1.0 / (k_safe - n))
        return 1.0 / (1.0 + jnp.exp(z))

    w = jnp.ones(N)

    # Left taper
    mask_left = x < n_left
    w = jnp.where(mask_left, _sigmoid(x, n_left), w)
    w = jnp.where(x == 0, 0.0, w)

    # Right taper
    # We want the signal to decay to 0 at index n_valid - 1
    # k should go from n_right to 0 as x goes from (n_valid - n_right) to n_valid

    # k = n_valid - x
    # at x = n_valid, k=0 -> w=0
    # at x = n_valid - n_right, k=n_right -> w=1

    k_right = n_valid - x
    mask_right = (x >= (n_valid - n_right)) & (x < n_valid)

    # We use k_right - 1 to align with the 0-based index logic if needed,
    # but let's stick to the continuous mapping.
    # Actually, let's use the same logic as planck_taper but shifted.
    # In planck_taper: k_right = (N - 1) - x. At x=N-1, k=0.
    # Here: k_right = (n_valid - 1) - x. At x=n_valid-1, k=0.

    k_right = (n_valid - 1) - x
    w = jnp.where(mask_right, _sigmoid(k_right, n_right), w)

    # Zero out invalid region
    w = jnp.where(x >= n_valid, 0.0, w)

    return w


@jit
def pad_and_shift(
    time: Array, strain: Array, dt: float, t_buffer: float
) -> tuple[Array, Array]:
    """
    Pads the waveform with zeros at the start and extends the time array backwards.

    Parameters
    ----------
        time: Time array of shape (N,).
        strain: Strain array of shape (N,).
        dt: Time step.
        t_buffer: Duration of the buffer to add (in seconds).

    Returns
    -------
        time_padded: New time array.
        strain_padded: New strain array with zero-padding at the start.
    """
    n_buffer = jnp.ceil(t_buffer / dt).astype(int)

    # Pad strain with zeros at the start
    strain_padded = jnp.pad(strain, (n_buffer, 0), mode="constant")

    # Extend time array backwards
    t_start = time[0]
    t_pre = t_start - jnp.arange(n_buffer, 0, -1) * dt
    time_padded = jnp.concatenate([t_pre, time])

    return time_padded, strain_padded


@jit
def condition_polarizations(
    time: Array,
    h_plus: Array,
    h_cross: Array,
    dt: float,
    mask: Array | None = None,
    t_taper_start: float = 1.0,
    t_taper_end: float = 1.0,
    t_buffer: float = 10.0,
    pad_to_next_pow2: bool = True,
) -> tuple[Array, Array, Array]:
    """
    Conditions the waveform for FFT:
    1. Tapers the start and end of the original signal.
    2. Adds a zero-buffer at the start.
    3. Optionally pads the end to the next power of 2.

    Parameters
    ----------
        time: Time array.
        h_plus: Plus polarization strain array.
        h_cross: Cross polarization strain array.
        dt: Time step.
        mask: Optional boolean mask indicating valid data points.
              If provided, the end taper is applied relative to the valid length.
        t_taper_start: Duration of the start taper (on the signal).
        t_taper_end: Duration of the end taper.
        t_buffer: Duration of the zero-buffer to add at the start.
        pad_to_next_pow2: Whether to pad the total length to the next power of 2.

    Returns
    -------
        time_cond: Conditioned time array.
        h_plus_cond: Conditioned plus polarization strain array.
        h_cross_cond: Conditioned cross polarization strain array.
    """
    N = h_plus.shape[-1]
    n_taper_start = int(t_taper_start / dt)
    n_taper_end = int(t_taper_end / dt)

    # 1. Apply taper to the original signal
    if mask is not None:
        n_valid = jnp.sum(mask).astype(int)
        window = planck_taper_masked(N, n_valid, n_taper_start, n_taper_end)
    else:
        window = planck_taper(N, n_taper_start, n_taper_end)

    h_plus_tapered = h_plus * window
    h_cross_tapered = h_cross * window

    # 2. Add buffer at the start
    time_buffered, h_plus_buffered = pad_and_shift(time, h_plus_tapered, dt, t_buffer)
    _, h_cross_buffered = pad_and_shift(time, h_cross_tapered, dt, t_buffer)

    # 3. Pad to next power of 2 (at the end)
    if pad_to_next_pow2:
        N_curr = h_plus_buffered.shape[-1]
        N_next_pow2 = 2 ** jnp.ceil(jnp.log2(N_curr)).astype(int)
        n_pad_end = N_next_pow2 - N_curr

        h_plus_cond = jnp.pad(h_plus_buffered, (0, n_pad_end), mode="constant")
        h_cross_cond = jnp.pad(h_cross_buffered, (0, n_pad_end), mode="constant")

        # Extend time array forwards
        t_end = time_buffered[-1]
        t_post = t_end + jnp.arange(1, n_pad_end + 1) * dt
        time_cond = jnp.concatenate([time_buffered, t_post])
    else:
        h_plus_cond = h_plus_buffered
        h_cross_cond = h_cross_buffered
        time_cond = time_buffered

    return time_cond, h_plus_cond, h_cross_cond
