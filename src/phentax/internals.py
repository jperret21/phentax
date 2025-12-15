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

from . import fits
from .constants import MRSUN_SI, MTSUN_SI, PC_SI

# =============================================================================
# Data structures (pytree-compatible)
# =============================================================================


class WaveformParams(NamedTuple):
    """
    Physical parameters for waveform generation.

    All masses in solar masses, spins dimensionless, distances in Mpc.
    """

    m1: float  # Primary mass (M_sun)
    m2: float  # Secondary mass (M_sun)
    s1z: float  # Primary spin z-component (dimensionless)
    s2z: float  # Secondary spin z-component (dimensionless)
    distance: float  # Luminosity distance (Mpc)
    inclination: float  # Inclination angle (radians)
    phi_ref: float  # Reference phase (radians)
    f_ref: float  # Reference frequency (Hz), or 0 for peak


class DerivedParams(NamedTuple):
    """
    Derived quantities computed from physical parameters.

    These are intermediate quantities used throughout the waveform computation.
    """

    M_total: float  # Total mass (M_sun)
    eta: float  # Symmetric mass ratio
    delta: float  # Mass difference ratio (m1-m2)/M
    chi_eff: float  # Effective spin
    chi1: float  # Spin 1 z-component
    chi2: float  # Spin 2 z-component
    Mf: float  # Final mass (solar masses)
    af: float  # Final spin (dimensionless)
    M_sec: float  # Total mass in seconds
    amp_0: float  # Amplitude prefactor


class PhaseCoeffs(NamedTuple):
    """
    Phase/omega calibration coefficients for a single mode.

    Contains all the collocation point values and region boundaries
    needed to construct the IMR phase.
    """

    # Ringdown quantities
    omega_ring: float  # QNM ringdown frequency
    gamma: float  # QNM damping rate
    gamma_n2: float  # Second overtone damping

    # Peak frequency
    omega_peak: float

    # Inspiral calibration
    t0_factor: float  # t0 calibration
    omega_cp_insp: float  # Inspiral collocation point

    # Intermediate calibration
    omega_cp_int1: float  # Intermediate CP 1
    omega_cp_int2: float  # Intermediate CP 2

    # Ringdown derivatives
    d2: float
    d3: float

    # Time shift
    tshift: float


class AmpCoeffs(NamedTuple):
    """
    Amplitude calibration coefficients for a single mode.

    Contains collocation point values and ringdown parameters.
    """

    # Inspiral
    amp_cp_insp: float

    # Intermediate
    amp_cp_int1: float
    amp_cp_int2: float

    # Peak
    amp_peak: float

    # Ringdown
    c3: float  # Overtone mixing
    gamma: float  # QNM damping
    gamma_n2: float  # Second overtone damping


class ModeCoeffs(NamedTuple):
    """
    Combined phase and amplitude coefficients for a single mode.
    """

    phase: PhaseCoeffs
    amp: AmpCoeffs
    mode: int  # Mode key (22, 21, 33, 44, 55, 20)


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
    m1, m2 = params.m1, params.m2
    s1z, s2z = params.s1z, params.s2z
    distance = params.distance

    # Ensure m1 >= m2
    m1, m2 = max(m1, m2), min(m1, m2)

    M_total = m1 + m2
    eta = m1 * m2 / (M_total * M_total)
    delta = (m1 - m2) / M_total

    # Effective spin
    chi_eff = (m1 * s1z + m2 * s2z) / M_total

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
# Coefficient computation functions
# =============================================================================


@jax.jit
def compute_phase_coeffs_22(derived: DerivedParams) -> PhaseCoeffs:
    """
    Compute phase coefficients for the 22 mode.

    Parameters
    ----------
    derived : DerivedParams
        Derived physical parameters.

    Returns
    -------
    PhaseCoeffs
        Phase calibration coefficients.
    """
    eta = derived.eta
    s1z = derived.chi1
    s2z = derived.chi2
    af = derived.af

    # Ringdown frequencies
    omega_ring = 2.0 * jnp.pi * fits.fring_22(af)
    gamma = 2.0 * jnp.pi * fits.fdamp_22(af)
    gamma_n2 = 2.0 * jnp.pi * fits.fdamp_n2_22(af)

    # Peak frequency (calibrated)
    omega_peak_fit = fits.peak_freq_22(eta, s1z, s2z)
    omega_peak = omega_ring * omega_peak_fit

    # Inspiral t0 calibration
    t0_factor = fits.inspiral_t0_22(eta, s1z, s2z)

    # Inspiral collocation point
    omega_cp_insp_fit = fits.inspiral_freq_cp_22(eta, s1z, s2z)
    omega_cp_insp = (
        omega_peak * omega_cp_insp_fit * 0.018
    )  # Approximate frequency scaling

    # Intermediate collocation points
    omega_cp_int1_fit = fits.intermediate_freq_cp1_22(eta, s1z, s2z)
    omega_cp_int2_fit = fits.intermediate_freq_cp2_22(eta, s1z, s2z)
    omega_cp_int1 = omega_peak * omega_cp_int1_fit * 0.5
    omega_cp_int2 = omega_peak * omega_cp_int2_fit * 0.75

    # Ringdown derivative corrections
    d2_fit = fits.rd_freq_d2_22(eta, s1z, s2z)
    d3_fit = fits.rd_freq_d3_22(eta, s1z, s2z)
    d2 = (omega_ring - omega_peak) * d2_fit * 0.1
    d3 = (omega_ring - omega_peak) * d3_fit * 0.01

    # Time shift
    tshift = fits.tshift_22(eta, s1z, s2z)

    return PhaseCoeffs(
        omega_ring=omega_ring,
        gamma=gamma,
        gamma_n2=gamma_n2,
        omega_peak=omega_peak,
        t0_factor=t0_factor,
        omega_cp_insp=omega_cp_insp,
        omega_cp_int1=omega_cp_int1,
        omega_cp_int2=omega_cp_int2,
        d2=d2,
        d3=d3,
        tshift=tshift,
    )


@jax.jit
def compute_amp_coeffs_22(derived: DerivedParams) -> AmpCoeffs:
    """
    Compute amplitude coefficients for the 22 mode.

    Parameters
    ----------
    derived : DerivedParams
        Derived physical parameters.

    Returns
    -------
    AmpCoeffs
        Amplitude calibration coefficients.
    """
    eta = derived.eta
    s1z = derived.chi1
    s2z = derived.chi2
    af = derived.af

    # Damping rates
    gamma = 2.0 * jnp.pi * fits.fdamp_22(af)
    gamma_n2 = 2.0 * jnp.pi * fits.fdamp_n2_22(af)

    # Calibrated amplitude factors
    amp_cp_insp_fit = fits.inspiral_amp_cp_22(eta, s1z, s2z)
    amp_cp_int1_fit = fits.intermediate_amp_cp1_22(eta, s1z, s2z)
    amp_cp_int2_fit = fits.intermediate_amp_cp2_22(eta, s1z, s2z)
    amp_peak_fit = fits.peak_amp_22(eta, s1z, s2z)
    c3_fit = fits.rd_amp_c3_22(eta, s1z, s2z)

    # Scale by eta and leading order
    amp_scale = derived.amp_0
    amp_cp_insp = amp_scale * amp_cp_insp_fit
    amp_cp_int1 = amp_scale * amp_cp_int1_fit
    amp_cp_int2 = amp_scale * amp_cp_int2_fit
    amp_peak = amp_scale * amp_peak_fit
    c3 = c3_fit * 0.1  # Mixing coefficient scaling

    return AmpCoeffs(
        amp_cp_insp=amp_cp_insp,
        amp_cp_int1=amp_cp_int1,
        amp_cp_int2=amp_cp_int2,
        amp_peak=amp_peak,
        c3=c3,
        gamma=gamma,
        gamma_n2=gamma_n2,
    )


def _compute_phase_coeffs_mode(derived: DerivedParams, mode: int) -> PhaseCoeffs:
    """
    Compute phase coefficients for a given mode.

    Parameters
    ----------
    derived : DerivedParams
        Derived physical parameters.
    mode : int
        Mode key (22, 21, 33, 44, 55, 20).

    Returns
    -------
    PhaseCoeffs
        Phase calibration coefficients.
    """
    eta = derived.eta
    s1z = derived.chi1
    s2z = derived.chi2
    af = derived.af

    # Ringdown frequencies (mode-dependent)
    omega_ring = 2.0 * jnp.pi * fits.fring(af, mode)
    gamma = 2.0 * jnp.pi * fits.fdamp(af, mode)
    gamma_n2 = 2.0 * jnp.pi * fits.fdamp_n2(af, mode)

    # Peak frequency (calibrated)
    omega_peak_fit = fits.peak_freq(eta, s1z, s2z, mode)
    omega_peak = omega_ring * omega_peak_fit

    # Inspiral calibration
    t0_factor = fits.inspiral_t0(eta, s1z, s2z, mode)
    omega_cp_insp_fit = fits.inspiral_freq_cp(eta, s1z, s2z, mode)
    omega_cp_insp = omega_peak * omega_cp_insp_fit * 0.018

    # Intermediate collocation points
    omega_cp_int1_fit = fits.intermediate_freq_cp1(eta, s1z, s2z, mode)
    omega_cp_int2_fit = fits.intermediate_freq_cp2(eta, s1z, s2z, mode)
    omega_cp_int1 = omega_peak * omega_cp_int1_fit * 0.5
    omega_cp_int2 = omega_peak * omega_cp_int2_fit * 0.75

    # Ringdown derivatives
    d2_fit = fits.rd_freq_d2(eta, s1z, s2z, mode)
    d3_fit = fits.rd_freq_d3(eta, s1z, s2z, mode)
    d2 = (omega_ring - omega_peak) * d2_fit * 0.1
    d3 = (omega_ring - omega_peak) * d3_fit * 0.01

    # Time shift
    tshift_val = fits.tshift(eta, s1z, s2z, mode)

    return PhaseCoeffs(
        omega_ring=omega_ring,
        gamma=gamma,
        gamma_n2=gamma_n2,
        omega_peak=omega_peak,
        t0_factor=t0_factor,
        omega_cp_insp=omega_cp_insp,
        omega_cp_int1=omega_cp_int1,
        omega_cp_int2=omega_cp_int2,
        d2=d2,
        d3=d3,
        tshift=tshift_val,
    )


def _compute_amp_coeffs_mode(derived: DerivedParams, mode: int) -> AmpCoeffs:
    """
    Compute amplitude coefficients for a given mode.

    Parameters
    ----------
    derived : DerivedParams
        Derived physical parameters.
    mode : int
        Mode key (22, 21, 33, 44, 55, 20).

    Returns
    -------
    AmpCoeffs
        Amplitude calibration coefficients.
    """
    eta = derived.eta
    s1z = derived.chi1
    s2z = derived.chi2
    af = derived.af

    # Damping rates
    gamma = 2.0 * jnp.pi * fits.fdamp(af, mode)
    gamma_n2 = 2.0 * jnp.pi * fits.fdamp_n2(af, mode)

    # Calibrated factors
    amp_cp_insp_fit = fits.inspiral_amp_cp(eta, s1z, s2z, mode)
    amp_cp_int1_fit = fits.intermediate_amp_cp1(eta, s1z, s2z, mode)
    amp_cp_int2_fit = fits.intermediate_amp_cp2(eta, s1z, s2z, mode)
    amp_peak_fit = fits.peak_amp(eta, s1z, s2z, mode)
    c3_fit = fits.rd_amp_c3(eta, s1z, s2z, mode)

    # Mode-dependent amplitude hierarchy
    ell, m = mode // 10, mode % 10
    mode_factor = jax.lax.cond(
        mode == 22,
        lambda: 1.0,
        lambda: jax.lax.cond(
            mode == 21,
            lambda: 0.4,
            lambda: jax.lax.cond(
                mode == 33,
                lambda: 0.44,
                lambda: jax.lax.cond(
                    mode == 44,
                    lambda: 0.21,
                    lambda: jax.lax.cond(
                        mode == 55,
                        lambda: 0.1,
                        lambda: jax.lax.cond(
                            mode == 20,
                            lambda: 0.05,
                            lambda: 1.0,
                        ),
                    ),
                ),
            ),
        ),
    )

    amp_scale = derived.amp_0 * mode_factor
    amp_cp_insp = amp_scale * amp_cp_insp_fit
    amp_cp_int1 = amp_scale * amp_cp_int1_fit
    amp_cp_int2 = amp_scale * amp_cp_int2_fit
    amp_peak = amp_scale * amp_peak_fit
    c3 = c3_fit * 0.1

    return AmpCoeffs(
        amp_cp_insp=amp_cp_insp,
        amp_cp_int1=amp_cp_int1,
        amp_cp_int2=amp_cp_int2,
        amp_peak=amp_peak,
        c3=c3,
        gamma=gamma,
        gamma_n2=gamma_n2,
    )


def compute_mode_coeffs(derived: DerivedParams, mode: int) -> ModeCoeffs:
    """
    Compute all coefficients for a single mode.

    Parameters
    ----------
    derived : DerivedParams
        Derived physical parameters.
    mode : int
        Mode key (22, 21, 33, 44, 55, 20).

    Returns
    -------
    ModeCoeffs
        Combined phase and amplitude coefficients.
    """
    phase_coeffs = _compute_phase_coeffs_mode(derived, mode)
    amp_coeffs = _compute_amp_coeffs_mode(derived, mode)
    return ModeCoeffs(phase=phase_coeffs, amp=amp_coeffs, mode=mode)


# =============================================================================
# Time grid generation
# =============================================================================


@jax.jit
def compute_time_grid(
    M_sec: float,
    f_low: float,
    dt: float,
    t_extra: float = 500.0,
) -> Tuple[jnp.ndarray, float, float]:
    """
    Compute time grid for waveform generation.

    Parameters
    ----------
    M_sec : float
        Total mass in seconds.
    f_low : float
        Starting frequency (Hz).
    dt : float
        Time step (seconds).
    t_extra : float
        Extra time after peak (in M).

    Returns
    -------
    times : array
        Time array (in M, centered on peak at t=0).
    t_start : float
        Start time.
    n_points : int
        Number of points.
    """
    # Convert f_low to dimensionless frequency
    omega_low = 2.0 * jnp.pi * f_low * M_sec

    # Estimate time to merger using Newtonian inspiral
    # t ~ 5/256 * M / eta * v^(-8) where v = (M omega / 2)^(1/3)
    # Simplified: t ~ (omega)^(-8/3)
    t_inspiral = 5.0 / (256.0 * omega_low ** (8.0 / 3.0))

    # Convert dt to dimensionless
    dt_M = dt / M_sec

    # Time grid (negative = before peak, positive = after)
    t_start = -t_inspiral
    t_end = t_extra

    n_points = jnp.int32((t_end - t_start) / dt_M)

    times = jnp.linspace(t_start, t_end, n_points)

    return times, t_start, n_points
