# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
Phase and omega coefficient computation for IMRPhenomT(HM).

This module implements the pPhase class functionality from phenomxpy,
computing all the coefficients needed for the IMR omega and phase ansatze.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from . import collocation, fits, pn_coeffs


class PhaseCoeffs22(NamedTuple):
    """
    All phase/omega coefficients for the 22 mode.

    Contains PN coefficients, pseudo-PN coefficients, ringdown parameters,
    and intermediate region coefficients.
    """

    # PN coefficients (TaylorT3)
    omega1PN: float
    omega1halfPN: float
    omega2PN: float
    omega2halfPN: float
    omega3PN: float
    omega3halfPN: float

    # Pseudo-PN coefficients (6 coefficients for omega inspiral)
    omegaInspC1: float
    omegaInspC2: float
    omegaInspC3: float
    omegaInspC4: float
    omegaInspC5: float
    omegaInspC6: float

    # Ringdown quantities
    omegaRING: float | Array  # 2*pi*fring
    alpha1RD: float | Array  # 2*pi*fdamp
    omegaRING_prec: float | Array  # For precessing case

    # Peak frequency
    omegaPeak: float

    # Ringdown ansatz coefficients
    c1: float
    c2: float
    c3: float
    c4: float
    c1_prec: float

    # Intermediate ansatz coefficients
    omegaMergerC1: float | Array
    omegaMergerC2: float | Array
    omegaMergerC3: float | Array

    # Intermediate region values
    omegaCut: float  # omega at inspiral cut
    domegaCut: float  # domega/dt at inspiral cut
    domegaPeak: float | Array  # domega/dt at peak

    # Times and cuts
    inspiral_cut: float  # tCut: transition time inspiral -> intermediate
    ringdown_cut: float  # = 0 (peak time)
    tt0: float | Array  # t0 from fit
    tEarly: float | Array  # Early time for phase offset

    # Phase continuity offsets
    phOffInsp: float | Array
    phOffMerger: float | Array
    phOffRD: float | Array

    # Powers of 5 for phase computation
    powers_of_5: jnp.ndarray


def compute_phase_coeffs_22(
    eta: float,
    chi1: float,
    chi2: float,
) -> PhaseCoeffs22:
    """
    Compute all phase/omega coefficients for the 22 mode.

    This function replicates the initialization of phenomxpy's pPhase class
    for the 22 mode.

    Parameters
    ----------
    eta : float
        Symmetric mass ratio.
    chi1, chi2 : float
        Dimensionless spin z-components.

    Returns
    -------
    PhaseCoeffs22
        All coefficients needed for phase/omega computation.
    """
    # Compute derived quantities
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * (1.0 + delta)
    m2 = 0.5 * (1.0 - delta)

    # Final state
    af = fits.final_spin_2017(eta, chi1, chi2)

    # Powers of 5 for phase computation
    base = jnp.power(5.0, 1.0 / 8.0)
    powers_of_5 = jnp.array(
        [1.0, base, base**2, base**3, base**4, base**5, base**6, base**7]
    )

    # Inspiral and ringdown cuts
    inspiral_cut = -26.982976386771437 / eta  # = -5 / (eta * 0.81^8)
    ringdown_cut = 0.0

    # ==============================
    # PN Coefficients
    # ==============================
    omega_pn = pn_coeffs.compute_omega_pn_coeffs(eta, chi1, chi2, delta, m1, m2)
    omega_pn_array = jnp.array(
        [
            omega_pn.omega1PN,
            omega_pn.omega1halfPN,
            omega_pn.omega2PN,
            omega_pn.omega2halfPN,
            omega_pn.omega3PN,
            omega_pn.omega3halfPN,
        ]
    )

    # ==============================
    # Collocation points and pseudo-PN coefficients
    # ==============================
    omega_cp_values, tt0, tEarly = collocation.compute_omega_collocation_points(
        eta, chi1, chi2, omega_pn_array
    )

    pseudo_pn = collocation.compute_omega_pseudo_pn_coeffs(
        omega_pn_array, omega_cp_values
    )

    # ==============================
    # Ringdown coefficients
    # ==============================
    # Important: fring must be divided by Mfinal, like in phenomxpy
    Mfinal = fits.final_mass_2017(eta, chi1, chi2)
    omegaRING = 2.0 * jnp.pi * fits.fring_22(af) / Mfinal
    alpha1RD = 2.0 * jnp.pi * fits.fdamp_22(af) / Mfinal
    omegaRING_prec = omegaRING  # Same for non-precessing

    omegaPeak = fits.peak_freq_22(eta, chi1, chi2)

    c2 = fits.rd_freq_d2_22(eta, chi1, chi2)
    c3 = fits.rd_freq_d3_22(eta, chi1, chi2)
    c4 = 0.0
    c1 = (1.0 + c3 + c4) * (omegaRING - omegaPeak) / c2 / (c3 + 2.0 * c4)
    c1_prec = c1  # Same for non-precessing

    # ==============================
    # Intermediate region coefficients
    # ==============================
    # Compute omega at inspiral cut using the inspiral ansatz
    omegaCut = _inspiral_ansatz_omega_single(
        inspiral_cut,
        eta,
        omega_pn_array,
        jnp.array(
            [
                pseudo_pn.c1,
                pseudo_pn.c2,
                pseudo_pn.c3,
                pseudo_pn.c4,
                pseudo_pn.c5,
                pseudo_pn.c6,
            ]
        ),
    )

    # Compute domega/dt at inspiral cut
    domegaCut = _inspiral_ansatz_domega(
        inspiral_cut,
        eta,
        omega_pn_array,
        jnp.array(
            [
                pseudo_pn.c1,
                pseudo_pn.c2,
                pseudo_pn.c3,
                pseudo_pn.c4,
                pseudo_pn.c5,
                pseudo_pn.c6,
            ]
        ),
    )

    # Compute domega/dt at peak (t=0) - this is the derivative of the ringdown ansatz
    # _ringdown_ansatz_domega returns the raw derivative, we normalize by omegaRING
    domegaPeak = -_ringdown_ansatz_domega(0.0, c1, c2, c3, c4) / omegaRING

    # Collocation point for intermediate region
    tcpMerger = -5.0 / (eta * jnp.power(0.95, 8))
    omegaMergerCP = 1.0 - fits.intermediate_freq_cp1_22(eta, chi1, chi2) / omegaRING
    omegaCutBar = 1.0 - omegaCut / omegaRING
    # Note: domegaCut from inspiral_ansatz_domega is already the derivative
    # phenomxpy: self.domegaCut = -numba_inspiral_ansatz_domega(...) / self.omegaRING
    domegaCut = -domegaCut / omegaRING

    # Solve 3x3 system for intermediate coefficients
    omegaMergerC1, omegaMergerC2, omegaMergerC3 = _solve_intermediate_omega_system(
        alpha1RD,
        inspiral_cut,
        tcpMerger,
        omegaCutBar,
        omegaMergerCP,
        domegaCut,
        domegaPeak,
        omegaPeak,
        omegaRING,
    )

    # ==============================
    # Phase continuity offsets
    # ==============================
    # These require computing phase values at boundaries
    # For now, compute the inspiral offset
    theta0 = collocation.THETA_COLLOCATION_POINTS[0]
    thetabarini = jnp.power(eta * (tt0 - tEarly), -1.0 / 8.0)

    pn_phase_at_thetabarini = _pn_ansatz_phase(
        thetabarini, eta, powers_of_5, omega_pn_array
    )
    inspiral_phase_at_tEarly = _inspiral_ansatz_phase_value(
        tEarly,
        eta,
        powers_of_5,
        omega_pn_array,
        jnp.array(
            [
                pseudo_pn.c1,
                pseudo_pn.c2,
                pseudo_pn.c3,
                pseudo_pn.c4,
                pseudo_pn.c5,
                pseudo_pn.c6,
            ]
        ),
        0.0,  # phOffInsp = 0 for this calculation
    )
    phOffInsp = pn_phase_at_thetabarini - inspiral_phase_at_tEarly

    # Merger offset: match inspiral and intermediate at inspiral_cut
    inspiral_phase_at_cut = _inspiral_ansatz_phase_value(
        inspiral_cut,
        eta,
        powers_of_5,
        omega_pn_array,
        jnp.array(
            [
                pseudo_pn.c1,
                pseudo_pn.c2,
                pseudo_pn.c3,
                pseudo_pn.c4,
                pseudo_pn.c5,
                pseudo_pn.c6,
            ]
        ),
        phOffInsp,
    )
    intermediate_phase_at_cut = _intermediate_ansatz_phase_value(
        inspiral_cut,
        alpha1RD,
        omegaMergerC1,
        omegaMergerC2,
        omegaMergerC3,
        omegaPeak,
        domegaPeak,
        omegaRING,
        0.0,
    )
    phOffMerger = inspiral_phase_at_cut - intermediate_phase_at_cut

    # Ringdown offset: match intermediate at t=0
    phOffRD = _intermediate_ansatz_phase_value(
        0.0,
        alpha1RD,
        omegaMergerC1,
        omegaMergerC2,
        omegaMergerC3,
        omegaPeak,
        domegaPeak,
        omegaRING,
        phOffMerger,
    )

    return PhaseCoeffs22(
        omega1PN=omega_pn.omega1PN,
        omega1halfPN=omega_pn.omega1halfPN,
        omega2PN=omega_pn.omega2PN,
        omega2halfPN=omega_pn.omega2halfPN,
        omega3PN=omega_pn.omega3PN,
        omega3halfPN=omega_pn.omega3halfPN,
        omegaInspC1=pseudo_pn.c1,
        omegaInspC2=pseudo_pn.c2,
        omegaInspC3=pseudo_pn.c3,
        omegaInspC4=pseudo_pn.c4,
        omegaInspC5=pseudo_pn.c5,
        omegaInspC6=pseudo_pn.c6,
        omegaRING=omegaRING,
        alpha1RD=alpha1RD,
        omegaRING_prec=omegaRING_prec,
        omegaPeak=omegaPeak,
        c1=c1,
        c2=c2,
        c3=c3,
        c4=c4,
        c1_prec=c1_prec,
        omegaMergerC1=omegaMergerC1,
        omegaMergerC2=omegaMergerC2,
        omegaMergerC3=omegaMergerC3,
        omegaCut=omegaCut,
        domegaCut=domegaCut,
        domegaPeak=domegaPeak,
        inspiral_cut=inspiral_cut,
        ringdown_cut=ringdown_cut,
        tt0=tt0,
        tEarly=tEarly,
        phOffInsp=phOffInsp,
        phOffMerger=phOffMerger,
        phOffRD=phOffRD,
        powers_of_5=powers_of_5,
    )


# =============================================================================
# Helper functions for ansatz evaluation
# =============================================================================


def _inspiral_ansatz_omega_single(
    time: float,
    eta: float,
    omega_pn_coeffs: jnp.ndarray,
    omega_pseudo_pn_coeffs: jnp.ndarray,
) -> float:
    """Evaluate inspiral omega ansatz at a single time."""
    theta = jnp.power(-eta * time / 5.0, -1.0 / 8.0)

    # TaylorT3 part
    taylort3 = collocation.pn_ansatz_omega(theta, omega_pn_coeffs)

    # Pseudo-PN part
    theta8 = jnp.power(theta, 8)
    theta9 = theta8 * theta
    theta10 = theta9 * theta
    theta11 = theta10 * theta
    theta12 = theta11 * theta
    theta13 = theta12 * theta

    fac = theta * theta * theta / 8.0
    pseudo_pn_sum = (
        omega_pseudo_pn_coeffs[0] * theta8
        + omega_pseudo_pn_coeffs[1] * theta9
        + omega_pseudo_pn_coeffs[2] * theta10
        + omega_pseudo_pn_coeffs[3] * theta11
        + omega_pseudo_pn_coeffs[4] * theta12
        + omega_pseudo_pn_coeffs[5] * theta13
    )

    return taylort3 + 2.0 * fac * pseudo_pn_sum


def _inspiral_ansatz_domega(
    time: float,
    eta: float,
    omega_pn_coeffs: jnp.ndarray,
    omega_pseudo_pn_coeffs: jnp.ndarray,
) -> float | Array:
    """Compute d(omega)/dt for inspiral ansatz at a single time."""
    # Use JAX autodiff
    return jax.grad(
        lambda t: _inspiral_ansatz_omega_single(
            t, eta, omega_pn_coeffs, omega_pseudo_pn_coeffs
        )
    )(time)


def _ringdown_ansatz_domega(
    time: float,
    c1: float,
    c2: float,
    c3: float,
    c4: float,
) -> float | Array:
    """Compute d(omega)/dt for ringdown ansatz at a single time."""
    expC = jnp.exp(c2 * time)
    expC2 = expC * expC

    num = c1 * c2 * c2 * expC * (4 * c4 * expC + c3 * (c4 + expC2))
    den = c4 + expC * (c3 + expC)
    return num / (den * den)


def _solve_intermediate_omega_system(
    alpha1RD: float,
    tCut: float,
    tcpMerger: float | Array,
    omegaCutBar: float,
    omegaMergerCP: float,
    domegaCut: float,
    domegaPeak: float,
    omegaPeak: float,
    omegaRING: float,
) -> tuple:
    """Solve 3x3 linear system for intermediate omega coefficients."""
    # System from Eqs. 28, 29, 31 in arXiv:2012.11923

    ascut = jnp.arcsinh(alpha1RD * tCut)
    ascut2 = ascut * ascut
    ascut3 = ascut * ascut2
    ascut4 = ascut * ascut3

    bascut = jnp.arcsinh(alpha1RD * tcpMerger)
    bascut2 = bascut * bascut
    bascut3 = bascut * bascut2
    bascut4 = bascut * bascut3

    dencut = jnp.sqrt(1.0 + tCut * tCut * alpha1RD * alpha1RD)

    # Build matrix and RHS
    matrix = jnp.zeros((3, 3))
    B = jnp.zeros(3)

    # Row 0: omega at tCut
    B = B.at[0].set(
        omegaCutBar - (1.0 - omegaPeak / omegaRING) - (domegaPeak / alpha1RD) * ascut
    )
    matrix = matrix.at[0, 0].set(ascut2)
    matrix = matrix.at[0, 1].set(ascut3)
    matrix = matrix.at[0, 2].set(ascut4)

    # Row 1: omega at tcpMerger
    B = B.at[1].set(
        omegaMergerCP - (1.0 - omegaPeak / omegaRING) - (domegaPeak / alpha1RD) * bascut
    )
    matrix = matrix.at[1, 0].set(bascut2)
    matrix = matrix.at[1, 1].set(bascut3)
    matrix = matrix.at[1, 2].set(bascut4)

    # Row 2: derivative at tCut
    B = B.at[2].set(domegaCut - domegaPeak / dencut)
    matrix = matrix.at[2, 0].set(2.0 * alpha1RD * ascut / dencut)
    matrix = matrix.at[2, 1].set(3.0 * alpha1RD * ascut2 / dencut)
    matrix = matrix.at[2, 2].set(4.0 * alpha1RD * ascut3 / dencut)

    # Solve
    solution = jnp.linalg.solve(matrix, B)

    return solution[0], solution[1], solution[2]


def _pn_ansatz_phase(
    thetabar: float | Array,
    eta: float | Array,
    powers_of_5: jnp.ndarray,
    omega_pn_coeffs: jnp.ndarray,
) -> float | Array:
    """Evaluate PN ansatz phase at thetabar."""
    # This is the integration of the PN omega ansatz
    # From phenomxpy numba_pn_ansatz_22_phase
    thetabar = thetabar * powers_of_5[1]
    thetabar2 = thetabar * thetabar
    thetabar3 = thetabar * thetabar2
    thetabar4 = thetabar * thetabar3
    thetabar5 = thetabar * thetabar4
    thetabar6 = thetabar * thetabar5
    thetabar7 = thetabar * thetabar6
    logthetabar = jnp.log(thetabar)

    aux = (
        1
        / eta
        / thetabar5
        * (
            -168
            - 280 * omega_pn_coeffs[0] * thetabar2
            - 420 * omega_pn_coeffs[1] * thetabar3
            - 840 * omega_pn_coeffs[2] * thetabar4
            + 840 * omega_pn_coeffs[3] * (logthetabar - 0.125 * jnp.log(5)) * thetabar5
            - 321 * thetabar6
            + 840 * omega_pn_coeffs[4] * thetabar6
            + 321 * logthetabar * thetabar6
            + 420 * omega_pn_coeffs[5] * thetabar7
        )
    ) / 84.0

    return aux


def _inspiral_ansatz_phase_value(
    time: float | Array,
    eta: float,
    powers_of_5: jnp.ndarray,
    omega_pn_coeffs: jnp.ndarray,
    omega_pseudo_pn_coeffs: jnp.ndarray,
    phOffInsp: float | Array,
) -> float | Array:
    """Evaluate inspiral ansatz phase at a single time."""

    thetabar = jnp.power(-eta * time, -1.0 / 8.0)
    thetabar2 = thetabar * thetabar
    thetabar3 = thetabar * thetabar2
    thetabar4 = thetabar * thetabar3
    thetabar5 = thetabar * thetabar4
    thetabar6 = thetabar * thetabar5
    thetabar7 = thetabar * thetabar6

    logmtime = jnp.log(-time)
    log_theta_bar = jnp.log(jnp.power(5, 0.125)) - 0.125 * (jnp.log(eta) + logmtime)

    aux = (
        -(
            1
            / powers_of_5[5]
            / (eta * eta)
            / time
            / thetabar7
            * (
                3 * (-107 + 280 * omega_pn_coeffs[4]) * powers_of_5[6]
                + 321 * log_theta_bar * powers_of_5[6]
                + 420 * omega_pn_coeffs[5] * thetabar * powers_of_5[7]
                + 56 * (25 * omega_pseudo_pn_coeffs[0] + 3 * eta * time) * thetabar2
                + 1050 * omega_pseudo_pn_coeffs[1] * powers_of_5[1] * thetabar3
                + 280
                * (3 * omega_pseudo_pn_coeffs[2] + eta * omega_pn_coeffs[0] * time)
                * powers_of_5[2]
                * thetabar4
                + 140
                * (5 * omega_pseudo_pn_coeffs[3] + 3 * eta * omega_pn_coeffs[1] * time)
                * powers_of_5[3]
                * thetabar5
                + 120
                * (5 * omega_pseudo_pn_coeffs[4] + 7 * eta * omega_pn_coeffs[2] * time)
                * powers_of_5[4]
                * thetabar6
                + 525 * omega_pseudo_pn_coeffs[5] * powers_of_5[5] * thetabar7
                + 105
                * eta
                * omega_pn_coeffs[3]
                * time
                * logmtime
                * powers_of_5[5]
                * thetabar7
            )
        )
        / 84.0
    )

    return aux + phOffInsp


def _intermediate_ansatz_phase_value(
    time: float,
    alpha1RD: float,
    omegaMergerC1: float,
    omegaMergerC2: float,
    omegaMergerC3: float,
    omegaPeak: float,
    domegaPeak: float,
    omegaRING: float | Array,
    phOffMerger: float | Array,
) -> float | Array:
    """Evaluate intermediate ansatz phase at a single time."""
    x = jnp.arcsinh(alpha1RD * time)
    x2 = x * x
    x3 = x * x2
    x4 = x * x3
    term1 = jnp.sqrt(1.0 + (alpha1RD * alpha1RD) * time * time)

    aux = omegaRING * time * (
        1.0
        - (
            2.0 * omegaMergerC1
            + 24.0 * omegaMergerC3
            + (6.0 * omegaMergerC2 + domegaPeak / alpha1RD) * x
            + (1.0 - omegaPeak / omegaRING)
            + (omegaMergerC1 + 12.0 * omegaMergerC3) * x2
            + omegaMergerC2 * x3
            + omegaMergerC3 * x4
        )
    ) - (omegaRING / alpha1RD) * term1 * (
        -domegaPeak / alpha1RD
        - 6.0 * omegaMergerC2
        - x * (2.0 * omegaMergerC1 + 24.0 * omegaMergerC3)
        - 3.0 * omegaMergerC2 * x2
        - 4.0 * omegaMergerC3 * x3
    )

    return aux + phOffMerger
