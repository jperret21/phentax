# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
Internal data structures and coefficient computation for IMRPhenomT(HM).

Contains WaveformParams, PhaseCoeffs, AmpCoeffs as frozen dataclasses,
and functions to compute calibration coefficients from physical parameters.
"""

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..utils.constants import MRSUN_SI, MTSUN_SI, PC_SI
from ..utils.utility import m1ofeta, m2ofeta
from . import fits

# =============================================================================
# Data structures (pytree-compatible)
# =============================================================================


class WaveformParams(NamedTuple):
    """
    Physical parameters for waveform generation.

    All masses in solar masses, spins dimensionless, distances in Mpc.
    """

    m1: float | Array  # Primary mass (M_sun)
    m2: float | Array  # Secondary mass (M_sun)
    s1z: float | Array  # Primary spin z-component (dimensionless)
    s2z: float | Array  # Secondary spin z-component (dimensionless)
    distance: float | Array  # Luminosity distance (Mpc)
    inclination: float | Array  # Inclination angle (radians)
    phi_ref: float | Array  # Reference phase (radians)
    f_ref: float | Array  # Reference frequency (Hz), or 0 for peak
    f_min: float | Array  # Minimum frequency (Hz)


class DerivedParams(NamedTuple):
    """
    Derived quantities computed from physical parameters.

    These are intermediate quantities used throughout the waveform computation.
    """

    m1: float | Array  # Primary mass (dimensionless)
    m2: float | Array  # Secondary mass (dimensionless)
    M: float | Array  # Total mass (dimensionless)
    m1_dim: float | Array  # Primary mass (M_sun)
    m2_dim: float | Array  # Secondary mass (M_sun)
    M_total: float | Array  # Total mass (M_sun)
    eta: float | Array  # Symmetric mass ratio
    delta: float | Array  # Mass difference ratio (m1-m2)/M
    chi_eff: float | Array  # Effective spin
    chi1: float | Array  # Spin 1 z-component
    chi2: float | Array  # Spin 2 z-component
    Mf: float | Array  # Final mass (solar masses)
    af: float | Array  # Final spin (dimensionless)
    M_sec: float | Array  # Total mass in seconds
    amp_0: float | Array  # Amplitude prefactor
    t_ref: float | Array = 0.0  # Reference time (seconds), default 0.0


# =============================================================================
# Parameter validation and derived quantities
# =============================================================================


@jax.jit
def compute_derived_params(params: WaveformParams) -> DerivedParams:
    """
    Compute derived quantities from physical parameters.

    Parameters
    ----------
    params : WaveformParams
        Input physical parameters.

    Returns
    -------
    DerivedParams
        Derived quantities for waveform computation.
    """
    m1_dim, m2_dim = params.m1, params.m2
    s1z, s2z = params.s1z, params.s2z
    distance = params.distance

    # compute dimensionless masses

    # Ensure m1 >= m2
    mass1 = jnp.maximum(m1_dim, m2_dim)
    mass2 = jnp.minimum(m1_dim, m2_dim)

    M_total = mass1 + mass2
    eta = mass1 * mass2 / (M_total * M_total)
    delta = (mass1 - mass2) / M_total

    # dimensionless masses
    m1 = m1ofeta(eta)
    m2 = m2ofeta(eta)
    M = m1 + m2

    # Effective spin
    chi_eff = (mass1 * s1z + mass2 * s2z) / M_total
    # Final state
    Mf_frac = fits.final_mass_2017(eta, s1z, s2z)
    af = fits.final_spin_2017(eta, s1z, s2z)
    Mf = Mf_frac * M_total

    # Total mass in seconds
    M_sec = M_total * MTSUN_SI

    # Amplitude prefactor: (G M / c^2) / D * (eta)
    # In geometric units: M * eta / D
    # Convert distance from Mpc to meters
    D_m = distance * 1e6 * PC_SI
    amp_0 = M_total * MRSUN_SI * eta / D_m

    return DerivedParams(
        m1=m1,
        m2=m2,
        M=M,
        m1_dim=mass1,
        m2_dim=mass2,
        M_total=M_total,
        eta=eta,
        delta=delta,
        chi_eff=chi_eff,
        chi1=s1z,
        chi2=s2z,
        Mf=Mf,
        af=af,
        M_sec=M_sec,
        amp_0=amp_0,
    )


# =============================================================================
# Time grid generation
# =============================================================================


@jax.jit
def compute_time_grid(
    M_sec: float | Array,
    f_min: float | Array,
    dt: float | Array,
    t_extra: float | Array = 500.0,
) -> Tuple[jnp.ndarray, float | Array, float | Array]:
    """
    Compute time grid for waveform generation.

    Parameters
    ----------
    M_sec : float | Array
        Total mass in seconds.
    f_min : float | Array
        Starting frequency (Hz).
    dt : float | Array
        Time step (seconds).
    t_extra : float | Array
        Extra time after peak (in M).

    Returns
    -------
    times : array
        Time array (in M, centered on peak at t=0).
    t_start : float | Array
        Start time.
    n_points : int
        Number of points.
    """
    # Convert f_min to dimensionless frequency
    omega_min = 2.0 * jnp.pi * f_min * M_sec

    # Estimate time to merger using Newtonian inspiral
    # t ~ 5/256 * M / eta * v^(-8) where v = (M omega / 2)^(1/3)
    # Simplified: t ~ (omega)^(-8/3)
    t_inspiral = 5.0 / (256.0 * omega_min ** (8.0 / 3.0))

    # Convert dt to dimensionless
    dt_M = dt / M_sec

    # Time grid (negative = before peak, positive = after)
    t_start = -t_inspiral
    t_end = t_extra

    n_points = jnp.int32((t_end - t_start) / dt_M)

    times = jnp.linspace(t_start, t_end, n_points)

    return times, t_start, n_points
