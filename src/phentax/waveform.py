# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
Main waveform generation API for IMRPhenomT(HM).

Provides the functional API for computing gravitational waveforms:
- compute_hlm: Single mode strain
- compute_hlms: Multiple modes
- compute_polarizations: h_plus and h_cross
"""

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from .ansatze import (
    imr_amplitude,
    imr_omega,
    inspiral_omega_taylort3,
    phase_from_omega_trapz,
    ringdown_amp_ansatz,
    ringdown_omega_ansatz,
)
from .internals import (
    AmpCoeffs,
    DerivedParams,
    ModeCoeffs,
    PhaseCoeffs,
    WaveformParams,
    compute_amp_coeffs_22,
    compute_derived_params,
    compute_mode_coeffs,
    compute_phase_coeffs_22,
)

# =============================================================================
# Helper functions
# =============================================================================


def make_params(
    m1: float,
    m2: float,
    s1z: float = 0.0,
    s2z: float = 0.0,
    distance: float = 100.0,
    inclination: float = 0.0,
    phi_ref: float = 0.0,
    f_ref: float = 0.0,
) -> WaveformParams:
    """
    Create a WaveformParams object with physical parameters.

    Parameters
    ----------
    m1 : float
        Primary mass in solar masses.
    m2 : float
        Secondary mass in solar masses.
    s1z : float, optional
        Primary spin z-component (dimensionless), default 0.
    s2z : float, optional
        Secondary spin z-component (dimensionless), default 0.
    distance : float, optional
        Luminosity distance in Mpc, default 100.
    inclination : float, optional
        Inclination angle in radians, default 0 (face-on).
    phi_ref : float, optional
        Reference phase in radians, default 0.
    f_ref : float, optional
        Reference frequency in Hz, default 0 (use peak).

    Returns
    -------
    WaveformParams
        Parameter container for waveform generation.
    """
    return WaveformParams(
        m1=m1,
        m2=m2,
        s1z=s1z,
        s2z=s2z,
        distance=distance,
        inclination=inclination,
        phi_ref=phi_ref,
        f_ref=f_ref,
    )


# =============================================================================
# Single mode computation
# =============================================================================


@jax.jit
def _compute_omega_22(
    times: jnp.ndarray,
    derived: DerivedParams,
    phase_coeffs: PhaseCoeffs,
) -> jnp.ndarray:
    """
    Compute omega(t) for the 22 mode.

    Parameters
    ----------
    times : array
        Time array in units of M (centered on peak at t=0).
    derived : DerivedParams
        Derived physical parameters.
    phase_coeffs : PhaseCoeffs
        Phase calibration coefficients.

    Returns
    -------
    array
        Omega values (dimensionless).
    """
    eta = derived.eta
    delta = derived.delta
    chi1 = derived.chi1
    chi2 = derived.chi2

    omega_ring = phase_coeffs.omega_ring
    omega_peak = phase_coeffs.omega_peak
    gamma = phase_coeffs.gamma
    d2 = phase_coeffs.d2
    d3 = phase_coeffs.d3

    # Identify regions
    t_peak = 0.0

    # Inspiral: t < t_insp_end (some time before peak)
    t_insp_end = -50.0 * derived.M_sec  # Approximate

    # Intermediate: t_insp_end < t < t_int_end
    t_int_end = -5.0 * derived.M_sec

    # TaylorT3 inspiral
    # theta = (eta/5 * (t_peak - t))^(-1/8)
    t_offset = jnp.maximum(t_peak - times, 1e-10)
    theta = jnp.power(eta / 5.0 * t_offset, -1.0 / 8.0)

    omega_insp = inspiral_omega_taylort3(theta, eta, chi1, chi2, delta)

    # Simple intermediate interpolation
    tau = (times - t_insp_end) / (t_int_end - t_insp_end + 1e-10)
    tau = jnp.clip(tau, 0.0, 1.0)
    omega_int = omega_insp * (1.0 - tau) + omega_peak * tau

    # Ringdown
    omega_rd = ringdown_omega_ansatz(
        times, t_peak, omega_peak, omega_ring, gamma, d2, d3
    )

    # Combine with smooth transitions
    omega = imr_omega(
        times, omega_insp, omega_int, omega_rd, t_insp_end, t_int_end, 2.0
    )

    # Ensure positivity
    omega = jnp.maximum(omega, 1e-10)

    return omega


@jax.jit
def _compute_amp_22(
    times: jnp.ndarray,
    omega: jnp.ndarray,
    derived: DerivedParams,
    amp_coeffs: AmpCoeffs,
) -> jnp.ndarray:
    """
    Compute amplitude(t) for the 22 mode.

    Parameters
    ----------
    times : array
        Time array in units of M.
    omega : array
        Frequency array.
    derived : DerivedParams
        Derived physical parameters.
    amp_coeffs : AmpCoeffs
        Amplitude calibration coefficients.

    Returns
    -------
    array
        Amplitude values.
    """
    t_peak = 0.0
    t_insp_end = -50.0 * derived.M_sec
    t_int_end = -5.0 * derived.M_sec

    amp_peak = amp_coeffs.amp_peak
    gamma = amp_coeffs.gamma
    gamma_n2 = amp_coeffs.gamma_n2
    c3 = amp_coeffs.c3

    # Inspiral amplitude: A ~ omega^(2/3)
    amp_insp = derived.amp_0 * jnp.power(omega, 2.0 / 3.0)

    # Intermediate: interpolate to peak
    tau = (times - t_insp_end) / (t_int_end - t_insp_end + 1e-10)
    tau = jnp.clip(tau, 0.0, 1.0)

    # Get inspiral amp at transition
    amp_at_int_start = derived.amp_0 * jnp.power(
        jnp.maximum(omega[jnp.argmin(jnp.abs(times - t_insp_end))], 1e-10), 2.0 / 3.0
    )

    amp_int = amp_at_int_start * (1.0 - tau**2) + amp_peak * tau**2

    # Ringdown
    amp_rd = ringdown_amp_ansatz(times, t_peak, amp_peak, gamma, gamma_n2, c3)

    # Combine
    amp = imr_amplitude(times, amp_insp, amp_int, amp_rd, t_insp_end, t_int_end, 2.0)

    # Ensure positivity
    amp = jnp.maximum(amp, 0.0)

    return amp


@jax.jit
def compute_hlm_22(
    times: jnp.ndarray,
    params: WaveformParams,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the 22 mode strain h_22(t).

    Parameters
    ----------
    times : array
        Time array in seconds.
    params : WaveformParams
        Physical parameters.

    Returns
    -------
    h_real : array
        Real part of h_22.
    h_imag : array
        Imaginary part of h_22.
    """
    # Compute derived quantities
    derived = compute_derived_params(params)

    # Convert times to dimensionless (in units of M)
    times_M = times / derived.M_sec

    # Compute coefficients
    phase_coeffs = compute_phase_coeffs_22(derived)
    amp_coeffs = compute_amp_coeffs_22(derived)

    # Compute omega and phase
    omega = _compute_omega_22(times_M, derived, phase_coeffs)
    dt_M = times_M[1] - times_M[0]
    phase = phase_from_omega_trapz(omega, dt_M, params.phi_ref)

    # Compute amplitude
    amp = _compute_amp_22(times_M, omega, derived, amp_coeffs)

    # Construct complex strain: h_lm = A * exp(-i * m * phi)
    m = 2  # For 22 mode
    h_real = amp * jnp.cos(-m * phase)
    h_imag = amp * jnp.sin(-m * phase)

    return h_real, h_imag


def compute_hlm(
    times: jnp.ndarray,
    params: WaveformParams,
    mode: int = 22,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute a single spherical harmonic mode h_lm(t).

    Parameters
    ----------
    times : array
        Time array in seconds.
    params : WaveformParams
        Physical parameters.
    mode : int, optional
        Mode to compute (22, 21, 33, 44, 55, 20). Default is 22.

    Returns
    -------
    h_real : array
        Real part of h_lm.
    h_imag : array
        Imaginary part of h_lm.

    Notes
    -----
    For modes other than 22, this currently uses simplified scaling
    from the 22 mode. Full higher-mode implementation coming soon.
    """
    if mode == 22:
        return compute_hlm_22(times, params)

    # For other modes, scale from 22 (simplified)
    h22_real, h22_imag = compute_hlm_22(times, params)

    # Mode hierarchy factors (approximate)
    mode_factors = {
        21: 0.4,
        33: 0.44,
        44: 0.21,
        55: 0.1,
        20: 0.05,
    }
    factor = mode_factors.get(mode, 1.0)

    # Adjust phase for m
    ell, m = mode // 10, mode % 10
    phase_factor = m / 2.0  # Approximate phase scaling

    return factor * h22_real, factor * h22_imag


# =============================================================================
# Multiple modes computation
# =============================================================================


def compute_hlms(
    times: jnp.ndarray,
    params: WaveformParams,
    modes: List[int] = [22],
    include_mode_20: bool = False,
) -> Dict[int, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Compute multiple spherical harmonic modes.

    Parameters
    ----------
    times : array
        Time array in seconds.
    params : WaveformParams
        Physical parameters.
    modes : list of int, optional
        Modes to compute. Default is [22].
    include_mode_20 : bool, optional
        Whether to include the (2,0) mode. Default is False.
        The (2,0) mode requires special treatment and is not
        included by default.

    Returns
    -------
    hlms : dict
        Dictionary mapping mode -> (h_real, h_imag).
    """
    hlms = {}

    for mode in modes:
        if mode == 20 and not include_mode_20:
            continue
        h_real, h_imag = compute_hlm(times, params, mode)
        hlms[mode] = (h_real, h_imag)

    return hlms


# =============================================================================
# Polarizations
# =============================================================================


@jax.jit
def compute_polarizations(
    times: jnp.ndarray,
    params: WaveformParams,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute the gravitational wave polarizations h_plus and h_cross.

    This is the main API function for generating IMRPhenomT waveforms.

    Parameters
    ----------
    times : array
        Time array in seconds.
    params : WaveformParams
        Physical parameters including masses, spins, distance, inclination.

    Returns
    -------
    h_plus : array
        Plus polarization of the gravitational wave strain.
    h_cross : array
        Cross polarization of the gravitational wave strain.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from phentax import make_params, compute_polarizations
    >>>
    >>> # Create parameters for a 30-30 solar mass binary
    >>> params = make_params(m1=30.0, m2=30.0, distance=100.0)
    >>>
    >>> # Generate time array
    >>> dt = 1.0 / 4096  # 4096 Hz sampling
    >>> times = jnp.arange(-1.0, 0.1, dt)
    >>>
    >>> # Compute waveform
    >>> hp, hc = compute_polarizations(times, params)
    """
    inclination = params.inclination

    # Compute 22 mode (dominant)
    h22_real, h22_imag = compute_hlm_22(times, params)

    # Spin-weighted spherical harmonics for (l=2, m=2) and (l=2, m=-2)
    # Use the simpler specialized functions
    from .utils import _swsh_2m2, _swsh_22

    Y22 = _swsh_22(inclination, 0.0)
    Y2m2 = _swsh_2m2(inclination, 0.0)

    # h = sum_lm h_lm * Y^{-2}_{lm}
    # h_plus = Re(h), h_cross = -Im(h)
    # For m>0: h_lm Y_lm + h_{l,-m} Y_{l,-m}
    # With h_{l,-m} = (-1)^l conj(h_{lm}) for real waveforms

    # h_22 contribution
    h22_complex = h22_real + 1j * h22_imag
    h2m2_complex = jnp.conj(h22_complex)  # h_{2,-2} = conj(h_{22})

    h_total = h22_complex * Y22 + h2m2_complex * Y2m2

    h_plus = jnp.real(h_total)
    h_cross = -jnp.imag(h_total)

    return h_plus, h_cross


def compute_polarizations_hm(
    times: jnp.ndarray,
    params: WaveformParams,
    modes: List[int] = [22, 21, 33, 44, 55],
    include_mode_20: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute polarizations including higher harmonics.

    Parameters
    ----------
    times : array
        Time array in seconds.
    params : WaveformParams
        Physical parameters.
    modes : list of int, optional
        Modes to include. Default is [22, 21, 33, 44, 55].
    include_mode_20 : bool, optional
        Whether to include the (2,0) mode. Default is False.

    Returns
    -------
    h_plus : array
        Plus polarization.
    h_cross : array
        Cross polarization.
    """
    from .utils import get_swsh

    inclination = params.inclination
    phi_ref = params.phi_ref  # Azimuthal angle (set to 0 for simplicity)

    # Compute all requested modes
    hlms = compute_hlms(times, params, modes, include_mode_20)

    # Initialize
    h_total = jnp.zeros_like(times, dtype=jnp.complex128)

    for mode, (h_real, h_imag) in hlms.items():
        ell = mode // 10
        m = mode % 10

        hlm = h_real + 1j * h_imag

        # Positive m contribution
        swsh_func = get_swsh(ell, m)
        Ylm = swsh_func(inclination, phi_ref)
        h_total = h_total + hlm * Ylm

        # Negative m contribution: h_{l,-m} = (-1)^l conj(h_{lm})
        if m != 0:
            swsh_func_m = get_swsh(ell, -m)
            Ylmm = swsh_func_m(inclination, phi_ref)
            hlmm = ((-1) ** ell) * jnp.conj(hlm)
            h_total = h_total + hlmm * Ylmm

    h_plus = jnp.real(h_total)
    h_cross = -jnp.imag(h_total)

    return h_plus, h_cross


# =============================================================================
# Convenience functions
# =============================================================================


def generate_waveform(
    m1: float,
    m2: float,
    s1z: float = 0.0,
    s2z: float = 0.0,
    distance: float = 100.0,
    inclination: float = 0.0,
    f_low: float = 20.0,
    sample_rate: float = 4096.0,
    duration: Optional[float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate a complete IMRPhenomT waveform.

    This is a convenience function that handles time grid generation
    and returns the waveform at the specified sample rate.

    Parameters
    ----------
    m1, m2 : float
        Component masses in solar masses.
    s1z, s2z : float, optional
        Dimensionless spin z-components. Default 0.
    distance : float, optional
        Luminosity distance in Mpc. Default 100.
    inclination : float, optional
        Inclination angle in radians. Default 0.
    f_low : float, optional
        Starting frequency in Hz. Default 20.
    sample_rate : float, optional
        Sample rate in Hz. Default 4096.
    duration : float, optional
        Duration in seconds. If None, computed from f_low.

    Returns
    -------
    times : array
        Time array in seconds.
    h_plus : array
        Plus polarization.
    h_cross : array
        Cross polarization.
    """
    params = make_params(
        m1=m1,
        m2=m2,
        s1z=s1z,
        s2z=s2z,
        distance=distance,
        inclination=inclination,
    )

    derived = compute_derived_params(params)

    dt = 1.0 / sample_rate

    if duration is None:
        # Estimate duration from f_low using Newtonian inspiral time
        M_sec = derived.M_sec
        omega_low = 2.0 * jnp.pi * f_low * M_sec
        # t ~ 5/256/eta * (omega)^(-8/3)
        t_inspiral = 5.0 / (256.0 * derived.eta) * jnp.power(omega_low, -8.0 / 3.0)
        t_inspiral_sec = float(t_inspiral * M_sec)
        duration = t_inspiral_sec + 0.5  # Add some buffer

    # Time array centered on merger
    n_samples = int(duration * sample_rate)
    times = jnp.linspace(-duration + 0.1, 0.1, n_samples)

    h_plus, h_cross = compute_polarizations(times, params)

    return times, h_plus, h_cross
