# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
Ansatz functions for IMRPhenomT(HM) waveform model.

Contains the inspiral, intermediate, and ringdown ansatze for both
phase/omega and amplitude. All functions are JAX-compatible.
"""

import jax
import jax.numpy as jnp
from jax import lax

# =============================================================================
# Inspiral omega (frequency) ansatze
# =============================================================================


@jax.jit
def inspiral_omega_ansatz(
    theta: jnp.ndarray,
    theta_insp: float,
    theta_cp: float,
    omega_insp: float,
    omega_cp: float,
    omega_insp_deriv: float,
    omega_cp_deriv: float,
) -> jnp.ndarray:
    """
    Inspiral omega ansatz: cubic polynomial in theta.

    The ansatz smoothly interpolates omega between the inspiral start
    and the collocation point, matching both values and derivatives.

    Parameters
    ----------
    theta : array
        Time-related variable (typically related to (t_peak - t)^(1/8)).
    theta_insp : float
        Theta at inspiral start.
    theta_cp : float
        Theta at collocation point.
    omega_insp : float
        Omega at inspiral start.
    omega_cp : float
        Omega at collocation point.
    omega_insp_deriv : float
        d(omega)/d(theta) at inspiral start.
    omega_cp_deriv : float
        d(omega)/d(theta) at collocation point.

    Returns
    -------
    array
        Omega values at the input theta points.
    """
    # Compute cubic coefficients to match values and derivatives at endpoints
    dtheta = theta_cp - theta_insp
    domega = omega_cp - omega_insp

    # Hermite interpolation coefficients
    a0 = omega_insp
    a1 = omega_insp_deriv
    a2 = (3.0 * domega / dtheta - 2.0 * omega_insp_deriv - omega_cp_deriv) / dtheta
    a3 = (-2.0 * domega / dtheta + omega_insp_deriv + omega_cp_deriv) / (
        dtheta * dtheta
    )

    t_norm = theta - theta_insp
    return a0 + a1 * t_norm + a2 * t_norm**2 + a3 * t_norm**3


@jax.jit
def inspiral_omega_taylort3(
    theta: jnp.ndarray,
    eta: float,
    chi1: float,
    chi2: float,
    delta: float,
) -> jnp.ndarray:
    """
    TaylorT3 omega ansatz for inspiral.

    This is the PN (post-Newtonian) expansion of orbital frequency as a
    function of the dimensionless time variable theta = (eta/5 * (tpeak - t)/M)^(-1/8).

    Parameters
    ----------
    theta : array
        Dimensionless time variable.
    eta : float
        Symmetric mass ratio.
    chi1, chi2 : float
        Dimensionless z-component spins.
    delta : float
        Mass difference (m1 - m2) / M.

    Returns
    -------
    array
        Omega values (dimensionless).
    """
    eta2 = eta * eta
    eta3 = eta2 * eta
    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)

    theta2 = theta * theta
    theta3 = theta2 * theta
    theta4 = theta3 * theta
    theta5 = theta4 * theta
    theta6 = theta5 * theta
    theta7 = theta6 * theta

    # TaylorT3 coefficients (simplified, main terms)
    # These come from the PN expansion
    c0 = 1.0
    c2 = 743.0 / 2688.0 + 11.0 / 32.0 * eta
    c3 = -3.0 * jnp.pi / 10.0 + (
        113.0 / 160.0 * delta * chi_a + (113.0 / 160.0 - 19.0 / 40.0 * eta) * chi_s
    )
    c4 = (
        1855099.0 / 14450688.0
        + 56975.0 / 258048.0 * eta
        + 371.0 / 2048.0 * eta2
        - (
            81.0 / 32.0 * delta * chi_a * chi_s
            + 81.0 / 64.0 * (chi_a**2 + chi_s**2)
            - 81.0 / 64.0 * eta * (chi_a**2 - chi_s**2)
        )
    )
    c5 = (
        -7729.0 / 21504.0 * jnp.pi
        + 13.0 / 256.0 * jnp.pi * eta
        + delta * chi_a * (140419.0 / 192000.0 - 2251.0 / 3840.0 * eta)
        + chi_s * (140419.0 / 192000.0 - 6709.0 / 1920.0 * eta + 701.0 / 960.0 * eta2)
    )
    c6 = (
        -720817631400877.0 / 288412611379200.0
        + 53.0 / 200.0 * jnp.pi**2
        + 107.0 / 280.0 * jnp.euler_gamma
        - 25302017977.0 / 4161798144.0 * eta
        - 30913.0 / 1835008.0 * eta2
        + 451.0 / 2048.0 * eta3
    )
    c7 = (
        -188516689.0 / 173408256.0 * jnp.pi
        + 488825.0 / 516096.0 * jnp.pi * eta
        - 141769.0 / 516096.0 * jnp.pi * eta2
    )

    omega = (
        theta3
        / 8.0
        * (
            c0
            + c2 * theta2
            + c3 * theta3
            + c4 * theta4
            + c5 * theta5
            + c6 * theta6
            + c7 * theta7
        )
    )
    return omega


# =============================================================================
# Intermediate omega ansatze
# =============================================================================


@jax.jit
def intermediate_omega_ansatz(
    t: jnp.ndarray,
    t_int_start: float,
    t_int_end: float,
    omega_start: float,
    omega_end: float,
    omega_start_deriv: float,
    omega_end_deriv: float,
    omega_cp1: float,
    omega_cp2: float,
    t_cp1: float,
    t_cp2: float,
) -> jnp.ndarray:
    """
    Intermediate omega ansatz: quintic polynomial with collocation points.

    Matches values and derivatives at region boundaries, plus two interior
    collocation point values for additional calibration.

    Parameters
    ----------
    t : array
        Time values.
    t_int_start, t_int_end : float
        Time at start/end of intermediate region.
    omega_start, omega_end : float
        Omega at start/end.
    omega_start_deriv, omega_end_deriv : float
        d(omega)/dt at start/end.
    omega_cp1, omega_cp2 : float
        Omega at collocation points.
    t_cp1, t_cp2 : float
        Time of collocation points.

    Returns
    -------
    array
        Omega values.
    """
    # We solve for a 5th order polynomial c0 + c1*t + ... + c5*t^5
    # with 6 constraints: 2 boundary values, 2 boundary derivatives, 2 collocation values

    # Shift time to origin at t_int_start
    dt = t - t_int_start
    dt_end = t_int_end - t_int_start
    dt_cp1 = t_cp1 - t_int_start
    dt_cp2 = t_cp2 - t_int_start

    # Build and solve the linear system (pre-computed inverse would be faster)
    # For simplicity, use a polynomial that interpolates the boundary conditions
    # and uses the CPs to add corrections

    # Hermite basis for boundary conditions
    tau = dt / dt_end
    tau2 = tau * tau
    tau3 = tau2 * tau

    h00 = 2.0 * tau3 - 3.0 * tau2 + 1.0
    h10 = tau3 - 2.0 * tau2 + tau
    h01 = -2.0 * tau3 + 3.0 * tau2
    h11 = tau3 - tau2

    # Base Hermite interpolation
    omega_hermite = (
        h00 * omega_start
        + h10 * dt_end * omega_start_deriv
        + h01 * omega_end
        + h11 * dt_end * omega_end_deriv
    )

    # Compute residuals at collocation points and add corrections
    # (This is a simplified version - full implementation would solve the linear system)
    tau_cp1 = dt_cp1 / dt_end
    tau_cp2 = dt_cp2 / dt_end

    h00_cp1 = 2.0 * tau_cp1**3 - 3.0 * tau_cp1**2 + 1.0
    h10_cp1 = tau_cp1**3 - 2.0 * tau_cp1**2 + tau_cp1
    h01_cp1 = -2.0 * tau_cp1**3 + 3.0 * tau_cp1**2
    h11_cp1 = tau_cp1**3 - tau_cp1**2

    omega_hermite_cp1 = (
        h00_cp1 * omega_start
        + h10_cp1 * dt_end * omega_start_deriv
        + h01_cp1 * omega_end
        + h11_cp1 * dt_end * omega_end_deriv
    )

    h00_cp2 = 2.0 * tau_cp2**3 - 3.0 * tau_cp2**2 + 1.0
    h10_cp2 = tau_cp2**3 - 2.0 * tau_cp2**2 + tau_cp2
    h01_cp2 = -2.0 * tau_cp2**3 + 3.0 * tau_cp2**2
    h11_cp2 = tau_cp2**3 - tau_cp2**2

    omega_hermite_cp2 = (
        h00_cp2 * omega_start
        + h10_cp2 * dt_end * omega_start_deriv
        + h01_cp2 * omega_end
        + h11_cp2 * dt_end * omega_end_deriv
    )

    # Residuals
    r1 = omega_cp1 - omega_hermite_cp1
    r2 = omega_cp2 - omega_hermite_cp2

    # Correction basis functions (vanish at boundaries with zero derivative)
    # phi_i(tau) = tau^2 * (1-tau)^2 * (tau - tau_j) for i != j
    basis1_cp1 = tau_cp1**2 * (1 - tau_cp1) ** 2 * (tau_cp1 - tau_cp2)
    basis2_cp1 = tau_cp1**2 * (1 - tau_cp1) ** 2 * (tau_cp1 - tau_cp1)  # = 0
    basis1_cp2 = tau_cp2**2 * (1 - tau_cp2) ** 2 * (tau_cp2 - tau_cp2)  # = 0
    basis2_cp2 = tau_cp2**2 * (1 - tau_cp2) ** 2 * (tau_cp2 - tau_cp1)

    # Solve 2x2 system
    # [basis1_cp1, basis2_cp1] [a1]   [r1]
    # [basis1_cp2, basis2_cp2] [a2] = [r2]
    # Simplified: basis2_cp1 = 0, basis1_cp2 = 0
    a1 = r1 / (basis1_cp1 + 1e-30)
    a2 = r2 / (basis2_cp2 + 1e-30)

    # Apply correction
    basis1 = tau2 * (1 - tau) ** 2 * (tau - tau_cp2)
    basis2 = tau2 * (1 - tau) ** 2 * (tau - tau_cp1)

    return omega_hermite + a1 * basis1 + a2 * basis2


# =============================================================================
# Ringdown omega ansatze
# =============================================================================


@jax.jit
def ringdown_omega_ansatz(
    t: jnp.ndarray,
    t_peak: float,
    omega_peak: float,
    omega_ring: float,
    gamma: float,
    d2: float,
    d3: float,
) -> jnp.ndarray:
    """
    Ringdown omega ansatz.

    Models the frequency evolution during ringdown as an exponential
    approach to the ringdown frequency with corrections.

    Parameters
    ----------
    t : array
        Time values (t > t_peak for ringdown).
    t_peak : float
        Time of peak amplitude.
    omega_peak : float
        Frequency at peak.
    omega_ring : float
        QNM ringdown frequency.
    gamma : float
        QNM damping rate.
    d2, d3 : float
        Higher-order derivative corrections.

    Returns
    -------
    array
        Omega values.
    """
    dt = t - t_peak

    # Exponential damping envelope
    exp_term = jnp.exp(-gamma * dt)

    # Frequency approaches omega_ring from omega_peak
    delta_omega = omega_ring - omega_peak

    # Ansatz with derivative corrections
    omega = omega_ring - delta_omega * exp_term * (1.0 + d2 * dt + d3 * dt**2)

    return omega


# =============================================================================
# Phase from omega integration
# =============================================================================


@jax.jit
def phase_from_omega_trapz(
    omega: jnp.ndarray,
    dt: float,
    phi0: float = 0.0,
) -> jnp.ndarray:
    """
    Integrate omega to get phase using trapezoidal rule.

    Parameters
    ----------
    omega : array
        Frequency array.
    dt : float
        Time step.
    phi0 : float, optional
        Initial phase.

    Returns
    -------
    array
        Phase values.
    """
    # Trapezoidal integration
    omega_avg = 0.5 * (omega[:-1] + omega[1:])
    dphi = omega_avg * dt
    phi = jnp.concatenate([jnp.array([phi0]), phi0 + jnp.cumsum(dphi)])
    return phi


# =============================================================================
# Inspiral amplitude ansatze
# =============================================================================


@jax.jit
def inspiral_amp_pn(
    omega: jnp.ndarray,
    eta: float,
    mode: int = 22,
) -> jnp.ndarray:
    """
    Post-Newtonian amplitude for inspiral.

    Leading order amplitude scaling with PN corrections.

    Parameters
    ----------
    omega : array
        Dimensionless frequency.
    eta : float
        Symmetric mass ratio.
    mode : int
        Mode number (22, 21, 33, 44, 55).

    Returns
    -------
    array
        Amplitude values (dimensionless).
    """
    # v = (M * omega)^(1/3) for circular orbits
    v = jnp.cbrt(omega)
    v2 = v * v

    # Leading order amplitude ~ eta * v^2
    # For l=m=2 mode
    amp_lo = eta * v2

    # Mode-dependent prefactor
    ell, m = mode // 10, mode % 10

    # Approximate mode hierarchy factors
    mode_factor = lax.cond(
        mode == 22,
        lambda: 1.0,
        lambda: lax.cond(
            mode == 21,
            lambda: 0.5,
            lambda: lax.cond(
                mode == 33,
                lambda: 0.75,
                lambda: lax.cond(
                    mode == 44,
                    lambda: 0.5,
                    lambda: lax.cond(
                        mode == 55,
                        lambda: 0.3,
                        lambda: lax.cond(
                            mode == 20,
                            lambda: 0.1,
                            lambda: 1.0,
                        ),
                    ),
                ),
            ),
        ),
    )

    return mode_factor * amp_lo


@jax.jit
def inspiral_amp_ansatz(
    omega: jnp.ndarray,
    omega_insp: float,
    omega_cp: float,
    amp_insp: float,
    amp_cp: float,
    eta: float,
) -> jnp.ndarray:
    """
    Inspiral amplitude ansatz with PN scaling and calibration.

    Parameters
    ----------
    omega : array
        Frequency values.
    omega_insp : float
        Frequency at inspiral start.
    omega_cp : float
        Frequency at collocation point.
    amp_insp : float
        Amplitude at inspiral start.
    amp_cp : float
        Amplitude at collocation point.
    eta : float
        Symmetric mass ratio.

    Returns
    -------
    array
        Amplitude values.
    """
    # PN leading order scaling: A ~ omega^(2/3)
    amp_pn_scale = jnp.power(omega, 2.0 / 3.0)
    amp_pn_insp = jnp.power(omega_insp, 2.0 / 3.0)
    amp_pn_cp = jnp.power(omega_cp, 2.0 / 3.0)

    # Linear interpolation in log space between calibration points
    log_amp_insp = jnp.log(amp_insp + 1e-30)
    log_amp_cp = jnp.log(amp_cp + 1e-30)
    log_pn_insp = jnp.log(amp_pn_insp + 1e-30)
    log_pn_cp = jnp.log(amp_pn_cp + 1e-30)

    # Correction factor interpolation
    corr_insp = log_amp_insp - log_pn_insp
    corr_cp = log_amp_cp - log_pn_cp

    # Linear interpolation of correction in omega
    frac = (omega - omega_insp) / (omega_cp - omega_insp + 1e-30)
    frac = jnp.clip(frac, 0.0, 1.0)

    correction = corr_insp + frac * (corr_cp - corr_insp)

    return amp_pn_scale * jnp.exp(correction)


# =============================================================================
# Intermediate amplitude ansatze
# =============================================================================


@jax.jit
def intermediate_amp_ansatz(
    t: jnp.ndarray,
    t_int_start: float,
    t_int_end: float,
    amp_start: float,
    amp_end: float,
    amp_cp1: float,
    amp_cp2: float,
    t_cp1: float,
    t_cp2: float,
) -> jnp.ndarray:
    """
    Intermediate amplitude ansatz: polynomial interpolation with collocation.

    Parameters
    ----------
    t : array
        Time values.
    t_int_start, t_int_end : float
        Time at start/end of intermediate region.
    amp_start, amp_end : float
        Amplitude at start/end.
    amp_cp1, amp_cp2 : float
        Amplitude at collocation points.
    t_cp1, t_cp2 : float
        Time of collocation points.

    Returns
    -------
    array
        Amplitude values.
    """
    # Normalized time
    dt = t_int_end - t_int_start
    tau = (t - t_int_start) / dt
    tau_cp1 = (t_cp1 - t_int_start) / dt
    tau_cp2 = (t_cp2 - t_int_start) / dt

    # Lagrange interpolation through 4 points
    # Points: (0, amp_start), (tau_cp1, amp_cp1), (tau_cp2, amp_cp2), (1, amp_end)

    def lagrange_basis(tau_val, idx, nodes):
        """Compute Lagrange basis polynomial."""
        result = 1.0
        for j, node in enumerate(nodes):
            if j != idx:
                result = result * (tau_val - node) / (nodes[idx] - node + 1e-30)
        return result

    nodes = jnp.array([0.0, tau_cp1, tau_cp2, 1.0])
    values = jnp.array([amp_start, amp_cp1, amp_cp2, amp_end])

    # Vectorized Lagrange interpolation
    def interp_point(t_val):
        result = 0.0
        for i in range(4):
            basis = 1.0
            for j in range(4):
                if i != j:
                    basis = basis * (t_val - nodes[j]) / (nodes[i] - nodes[j] + 1e-30)
            result = result + values[i] * basis
        return result

    return jax.vmap(interp_point)(tau)


# =============================================================================
# Ringdown amplitude ansatze
# =============================================================================


@jax.jit
def ringdown_amp_ansatz(
    t: jnp.ndarray,
    t_peak: float,
    amp_peak: float,
    gamma: float,
    gamma_n2: float,
    c3: float,
) -> jnp.ndarray:
    """
    Ringdown amplitude ansatz.

    Models exponential decay with multiple QNM overtones.

    Parameters
    ----------
    t : array
        Time values (t > t_peak for ringdown).
    t_peak : float
        Time of peak amplitude.
    amp_peak : float
        Amplitude at peak.
    gamma : float
        Fundamental QNM damping rate.
    gamma_n2 : float
        Second overtone damping rate.
    c3 : float
        Mixing coefficient for overtones.

    Returns
    -------
    array
        Amplitude values.
    """
    dt = t - t_peak

    # Primary exponential decay
    exp_primary = jnp.exp(-gamma * dt)

    # Second overtone contribution
    exp_n2 = jnp.exp(-gamma_n2 * dt)

    # Combined decay with mixing
    amp = amp_peak * (exp_primary + c3 * (exp_n2 - exp_primary))

    return amp


# =============================================================================
# Full IMR amplitude combination
# =============================================================================


@jax.jit
def imr_amplitude(
    t: jnp.ndarray,
    amp_insp: jnp.ndarray,
    amp_int: jnp.ndarray,
    amp_rd: jnp.ndarray,
    t_insp_end: float,
    t_int_end: float,
    transition_width: float = 1.0,
) -> jnp.ndarray:
    """
    Combine inspiral, intermediate, and ringdown amplitudes smoothly.

    Uses hyperbolic tangent transitions between regions.

    Parameters
    ----------
    t : array
        Time values.
    amp_insp, amp_int, amp_rd : array
        Amplitude arrays for each region.
    t_insp_end : float
        End of inspiral / start of intermediate.
    t_int_end : float
        End of intermediate / start of ringdown.
    transition_width : float
        Width of tanh transitions.

    Returns
    -------
    array
        Combined amplitude.
    """
    # Transition functions
    sigma1 = 0.5 * (1.0 + jnp.tanh((t - t_insp_end) / transition_width))
    sigma2 = 0.5 * (1.0 + jnp.tanh((t - t_int_end) / transition_width))

    # Blend regions
    amp = (
        (1.0 - sigma1) * amp_insp + sigma1 * (1.0 - sigma2) * amp_int + sigma2 * amp_rd
    )

    return amp


# =============================================================================
# Full IMR omega combination
# =============================================================================


@jax.jit
def imr_omega(
    t: jnp.ndarray,
    omega_insp: jnp.ndarray,
    omega_int: jnp.ndarray,
    omega_rd: jnp.ndarray,
    t_insp_end: float,
    t_int_end: float,
    transition_width: float = 1.0,
) -> jnp.ndarray:
    """
    Combine inspiral, intermediate, and ringdown omega smoothly.

    Uses hyperbolic tangent transitions between regions.

    Parameters
    ----------
    t : array
        Time values.
    omega_insp, omega_int, omega_rd : array
        Omega arrays for each region.
    t_insp_end : float
        End of inspiral / start of intermediate.
    t_int_end : float
        End of intermediate / start of ringdown.
    transition_width : float
        Width of tanh transitions.

    Returns
    -------
    array
        Combined omega.
    """
    # Transition functions
    sigma1 = 0.5 * (1.0 + jnp.tanh((t - t_insp_end) / transition_width))
    sigma2 = 0.5 * (1.0 + jnp.tanh((t - t_int_end) / transition_width))

    # Blend regions
    omega = (
        (1.0 - sigma1) * omega_insp
        + sigma1 * (1.0 - sigma2) * omega_int
        + sigma2 * omega_rd
    )

    return omega
