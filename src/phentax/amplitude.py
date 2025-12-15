# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
Amplitude coefficient computation for IMRPhenomT(HM).

This module implements the pAmp class functionality from phenomxpy,
computing all the coefficients needed for the IMR amplitude ansatze.
"""

# todo: format in a more jaxic way. Now there are if statements that could be jitted away.
# todo: we should really vectorize over harmonic modes.

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from . import fits, phase


class AmplitudeCoeffs22(NamedTuple):
    """
    All amplitude coefficients for the 22 mode.
    """

    # PN coefficients
    ampN: float
    amp0halfPNreal: float
    amp0halfPNimag: float
    amp1PNreal: float
    amp1PNimag: float
    amp1halfPNreal: float | Array
    amp1halfPNimag: float
    amp2PNreal: float | Array
    amp2PNimag: float
    amp2halfPNreal: float
    amp2halfPNimag: float
    amp3PNreal: float
    amp3PNimag: float
    amp3halfPNreal: float
    amp3halfPNimag: float
    amplog: float

    # Pseudo-PN coefficients (3 coefficients)
    inspC1: float
    inspC2: float
    inspC3: float

    # Ringdown coefficients
    alpha1RD: float
    alpha2RD: float
    alpha21RD: float
    alpha1RD_prec: float
    alpha2RD_prec: float
    alpha21RD_prec: float

    ampPeak: float
    c1: float
    c2: float
    c3: float
    c4: float
    c1_prec: float
    c2_prec: float
    c4_prec: float

    # Intermediate coefficients
    mergerC1: float | Array
    mergerC2: float | Array
    mergerC3: float | Array
    mergerC4: float | Array

    # Cuts and other quantities
    inspiral_cut: float
    ringdown_cut: float
    tshift: float

    # Collocation points (stored for debugging/verification)
    ampInspCP1: float
    ampInspCP2: float
    ampInspCP3: float
    ampMergerCP1: float
    dampMECO: float | Array
    phiCutPNAMP: float


def compute_amplitude_coeffs_22(
    eta: float,
    chi1: float,
    chi2: float,
    phase_coeffs: phase.PhaseCoeffs22,
) -> AmplitudeCoeffs22:
    """
    Compute all amplitude coefficients for the 22 mode.

    Parameters
    ----------
    eta : float
        Symmetric mass ratio.
    chi1, chi2 : float
        Dimensionless spin z-components.
    phase_coeffs : PhaseCoeffs22
        Phase coefficients (needed for omega evaluation).

    Returns
    -------
    AmplitudeCoeffs22
        All coefficients needed for amplitude computation.
    """
    # Derived quantities
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * (1.0 + delta)
    m2 = 0.5 * (1.0 - delta)

    # Spins
    s1z = chi1
    s2z = chi2
    chis = 0.5 * (s1z + s2z)
    chia = 0.5 * (s1z - s2z)
    S0 = m1 * s1z + m2 * s2z

    eta2 = eta * eta
    eta3 = eta2 * eta

    # ==============================
    # PN Coefficients (Mode 22)
    # ==============================
    ampN = 1.0
    amp0halfPNreal = 0.0
    amp0halfPNimag = 0.0
    amp1PNreal = -107.0 / 42.0 + (55.0 * eta) / 42.0
    amp1PNimag = 0.0
    amp1halfPNreal = (
        (-4.0 * chis) / 3.0
        - (4.0 * chia * delta) / 3.0
        + (4.0 * chis * eta) / 3.0
        + 2.0 * jnp.pi
    )
    amp1halfPNimag = 0.0
    amp2PNreal = (
        -2173.0 / 1512.0 - (1069.0 * eta) / 216.0 + (2047.0 * eta2) / 1512.0 + S0**2
    )
    amp2PNimag = 0.0
    amp2halfPNreal = (-107.0 * jnp.pi) / 21.0 + (34.0 * eta * jnp.pi) / 21.0
    amp2halfPNimag = -24.0 * eta
    amp3PNreal = (
        27027409.0 / 646800.0
        - (278185.0 * eta) / 33264.0
        - (20261.0 * eta2) / 2772.0
        + (114635.0 * eta3) / 99792.0
        - (856.0 * 0.5772156649015329) / 105.0  # Euler gamma
        + (2.0 * jnp.pi**2) / 3.0
        + (41.0 * eta * jnp.pi**2) / 96.0
    )
    amp3PNimag = (428.0 * jnp.pi) / 105.0
    amp3halfPNreal = (
        (-2173.0 * jnp.pi) / 756.0
        - (2495.0 * eta * jnp.pi) / 378.0
        + (40.0 * eta2 * jnp.pi) / 27.0
    )
    amp3halfPNimag = (14333.0 * eta) / 162.0 - (4066.0 * eta2) / 945.0
    amplog = -428.0 / 105.0

    pn_real_coeffs = jnp.array(
        [
            ampN,
            amp0halfPNreal,
            amp1PNreal,
            amp1halfPNreal,
            amp2PNreal,
            amp2halfPNreal,
            amp3PNreal,
            amp3halfPNreal,
            amplog,
        ]
    )
    pn_imag_coeffs = jnp.array(
        [
            amp0halfPNimag,
            amp1PNimag,
            amp1halfPNimag,
            amp2PNimag,
            amp2halfPNimag,
            amp3PNimag,
            amp3halfPNimag,
        ]
    )

    # ==============================
    # Inspiral Coefficients
    # ==============================
    # Collocation points
    tinsppoints = jnp.array([-2000.0, -250.0, -150.0])
    ampInspCP1 = fits.inspiral_amp_cp_22(eta, s1z, s2z, 1)
    ampInspCP2 = fits.inspiral_amp_cp_22(eta, s1z, s2z, 2)
    ampInspCP3 = fits.inspiral_amp_cp_22(eta, s1z, s2z, 3)
    ampInspCP = jnp.array([ampInspCP1, ampInspCP2, ampInspCP3])

    # Solve for pseudo-PN coefficients
    inspC1, inspC2, inspC3 = _solve_inspiral_amplitude_system(
        tinsppoints, ampInspCP, eta, pn_real_coeffs, pn_imag_coeffs, phase_coeffs
    )

    # ==============================
    # Ringdown Coefficients
    # ==============================
    af = fits.final_spin_2017(eta, s1z, s2z)
    Mf = fits.final_mass_2017(eta, s1z, s2z)

    # Frequencies need to be divided by Mfinal
    fdamp = fits.fdamp_22(af) / Mf
    fdampn2 = fits.fdamp_n2_22(af) / Mf

    alpha1RD = 2.0 * jnp.pi * fdamp
    alpha2RD = 2.0 * jnp.pi * fdampn2
    alpha21RD = 0.5 * (alpha2RD - alpha1RD)

    # Precessing (same for aligned)
    alpha1RD_prec = alpha1RD
    alpha2RD_prec = alpha2RD
    alpha21RD_prec = alpha21RD

    ampPeak = fits.peak_amp_22(eta, s1z, s2z)
    c3 = fits.rd_amp_c3_22(eta, s1z, s2z)
    c2 = alpha21RD
    c2_prec = alpha21RD_prec

    # Adjust c2 if needed
    coshc3 = jnp.cosh(c3)
    tanhc3 = jnp.tanh(c3)

    limit = jnp.abs(0.5 * alpha1RD / tanhc3)
    c2 = jnp.where(c2 > limit, -limit, c2)

    limit_prec = jnp.abs(0.5 * alpha1RD_prec / tanhc3)
    c2_prec = jnp.where(c2_prec > limit_prec, -limit_prec, c2_prec)

    c1 = ampPeak * alpha1RD * coshc3 * coshc3 / c2
    c4 = ampPeak - c1 * tanhc3
    c1_prec = ampPeak * alpha1RD_prec * coshc3 * coshc3 / c2_prec
    c4_prec = ampPeak - c1_prec * tanhc3

    # ==============================
    # Intermediate Coefficients
    # ==============================
    inspiral_cut = -150.0
    # tshift is computed from fits.tshift_22 but we don't have it in fits.py yet?
    # phenomxpy: self.tshift = IMRPhenomT_tshift(self._pWF)
    # I need to check if tshift_22 is in fits.py
    # Assuming it is or I need to add it.
    # Let's check fits.py for tshift
    tshift = fits.tshift_22(eta, s1z, s2z)
    ringdown_cut = tshift

    ampMergerCP1 = fits.intermediate_amp_cp1_22(eta, s1z, s2z)
    tcpMerger = -25.0

    mergerC1, mergerC2, mergerC3, mergerC4, dampMECO = (
        _solve_intermediate_amplitude_system(
            inspiral_cut,
            tcpMerger,
            tshift,
            alpha1RD,
            ampPeak,
            ampMergerCP1,
            eta,
            pn_real_coeffs,
            pn_imag_coeffs,
            jnp.array([inspC1, inspC2, inspC3]),
            phase_coeffs,
        )
    )

    phiCutPNAMP = 0.0  # For 22 mode

    return AmplitudeCoeffs22(
        ampN=ampN,
        amp0halfPNreal=amp0halfPNreal,
        amp0halfPNimag=amp0halfPNimag,
        amp1PNreal=amp1PNreal,
        amp1PNimag=amp1PNimag,
        amp1halfPNreal=amp1halfPNreal,
        amp1halfPNimag=amp1halfPNimag,
        amp2PNreal=amp2PNreal,
        amp2PNimag=amp2PNimag,
        amp2halfPNreal=amp2halfPNreal,
        amp2halfPNimag=amp2halfPNimag,
        amp3PNreal=amp3PNreal,
        amp3PNimag=amp3PNimag,
        amp3halfPNreal=amp3halfPNreal,
        amp3halfPNimag=amp3halfPNimag,
        amplog=amplog,
        inspC1=inspC1,
        inspC2=inspC2,
        inspC3=inspC3,
        alpha1RD=alpha1RD,
        alpha2RD=alpha2RD,
        alpha21RD=alpha21RD,
        alpha1RD_prec=alpha1RD_prec,
        alpha2RD_prec=alpha2RD_prec,
        alpha21RD_prec=alpha21RD_prec,
        ampPeak=ampPeak,
        c1=c1,
        c2=c2,
        c3=c3,
        c4=c4,
        c1_prec=c1_prec,
        c2_prec=c2_prec,
        c4_prec=c4_prec,
        mergerC1=mergerC1,
        mergerC2=mergerC2,
        mergerC3=mergerC3,
        mergerC4=mergerC4,
        inspiral_cut=inspiral_cut,
        ringdown_cut=ringdown_cut,
        tshift=tshift,
        ampInspCP1=ampInspCP1,
        ampInspCP2=ampInspCP2,
        ampInspCP3=ampInspCP3,
        ampMergerCP1=ampMergerCP1,
        dampMECO=dampMECO,
        phiCutPNAMP=phiCutPNAMP,
    )


# =============================================================================
# Helper functions
# =============================================================================


def _pn_ansatz_amplitude(
    x: float | Array,
    fac0: float | Array,
    pn_real: jnp.ndarray,
    pn_imag: jnp.ndarray,
) -> Array:
    """
    Evaluate PN ansatz amplitude at x = (omega/2)^(2/3).

    Parameters
    ----------
    x : float
        PN parameter x.
    fac0 : float
        Prefactor 2 * eta * sqrt(16*pi/5).
    pn_real : jnp.ndarray
        Real PN coefficients [ampN, amp0halfPNreal, ..., amplog].
    pn_imag : jnp.ndarray
        Imaginary PN coefficients [amp0halfPNimag, ...].

    Returns
    -------
    Array
        PN amplitude.
    """
    xhalf = jnp.sqrt(x)
    x1half = x * xhalf
    x2 = x * x
    x2half = x2 * xhalf
    x3 = x2 * x
    x3half = x3 * xhalf

    # Real part
    # pn_real: [ampN, amp0half, amp1, amp1half, amp2, amp2half, amp3, amp3half, amplog]
    ampreal = (
        pn_real[0]
        + pn_real[1] * xhalf
        + pn_real[2] * x
        + pn_real[3] * x1half
        + pn_real[4] * x2
        + pn_real[5] * x2half
        + pn_real[6] * x3
        + pn_real[7] * x3half
        + pn_real[8] * jnp.log(16.0 * x) * x3
    )

    # Imaginary part
    # pn_imag: [amp0half, amp1, amp1half, amp2, amp2half, amp3, amp3half]
    ampimag = (
        pn_imag[0] * xhalf
        + pn_imag[1] * x
        + pn_imag[2] * x1half
        + pn_imag[3] * x2
        + pn_imag[4] * x2half
        + pn_imag[5] * x3
        + pn_imag[6] * x3half
    )

    return fac0 * x * (ampreal + 1j * ampimag)


def _inspiral_ansatz_amplitude(
    x: float | Array,
    fac0: float | Array,
    pn_real: jnp.ndarray,
    pn_imag: jnp.ndarray,
    pseudo_pn: jnp.ndarray,
) -> Array:
    """
    Evaluate inspiral ansatz amplitude (PN + pseudo-PN).

    Parameters
    ----------
    x : float
        PN parameter x.
    fac0 : float
        Prefactor.
    pn_real, pn_imag : jnp.ndarray
        PN coefficients.
    pseudo_pn : jnp.ndarray
        Pseudo-PN coefficients [inspC1, inspC2, inspC3].

    Returns
    -------
    Array
        Inspiral amplitude.
    """
    # PN part
    pn_amp = _pn_ansatz_amplitude(x, fac0, pn_real, pn_imag)

    # Pseudo-PN part (only affects real part)
    x2 = x * x
    x4 = x2 * x2
    x4half = x4 * jnp.sqrt(x)
    x5 = x4 * x

    pseudo_pn_term = pseudo_pn[0] * x4 + pseudo_pn[1] * x4half + pseudo_pn[2] * x5

    return pn_amp + fac0 * x * pseudo_pn_term


def _imr_omega(
    time: float, phase_coeffs: phase.PhaseCoeffs22, eta: float
) -> float | Array:
    """Evaluate IMR omega at a single time."""
    # Note: phase_coeffs.inspiral_cut is for phase, not amplitude

    # We need to handle the regions correctly.
    # pPhase.imr_omega uses pPhase.inspiral_cut and pPhase.ringdown_cut (which is 0)

    # For scalar time:
    # if time < inspiral_cut: inspiral_ansatz
    # elif time >= ringdown_cut: ringdown_ansatz (which is just omegaRING for omega?)
    # else: intermediate_ansatz

    # Wait, ringdown ansatz for omega is constant omegaRING?
    # No, ringdown ansatz for omega is defined in pPhase.
    # Let's check phase.py again.
    # It has _ringdown_ansatz_domega but not _ringdown_ansatz_omega?
    # Ah, phenomxpy pPhase.ringdown_ansatz_omega returns omegaRING * (1 - w) ... no that's intermediate.
    # pPhase.ringdown_ansatz_omega returns omegaRING?
    # Let's check phenomxpy again.

    # In phenomxpy internals.py:
    # def ringdown_ansatz_omega(self, times):
    #     ...
    #     return self.omegaRING_prec + self.c1_prec * self._ringdown_ansatz_domega(...)

    # So it's omegaRING + c1 * domega_rd

    # I need to implement this.

    # For now, let's assume we are in inspiral or intermediate for the amplitude calculation.
    # The amplitude calculation uses omega for PN ansatz (x = (omega/2)^(2/3)).
    # The collocation points are at -2000, -250, -150.
    # These are all < 0.
    # phase_coeffs.inspiral_cut is around -2000 or so.
    # So -150 is likely in intermediate region of phase.

    # So I need full IMR omega evaluation.

    # I'll implement _imr_omega properly.

    if time < phase_coeffs.inspiral_cut:
        # Inspiral
        omega_pn_coeffs = jnp.array(
            [
                phase_coeffs.omega1PN,
                phase_coeffs.omega1halfPN,
                phase_coeffs.omega2PN,
                phase_coeffs.omega2halfPN,
                phase_coeffs.omega3PN,
                phase_coeffs.omega3halfPN,
            ]
        )
        omega_pseudo_pn_coeffs = jnp.array(
            [
                phase_coeffs.omegaInspC1,
                phase_coeffs.omegaInspC2,
                phase_coeffs.omegaInspC3,
                phase_coeffs.omegaInspC4,
                phase_coeffs.omegaInspC5,
                phase_coeffs.omegaInspC6,
            ]
        )
        return phase._inspiral_ansatz_omega_single(
            time, eta, omega_pn_coeffs, omega_pseudo_pn_coeffs
        )

    elif time >= phase_coeffs.ringdown_cut:  # 0.0
        # Ringdown
        # omega = omegaRING + c1 * domega_rd
        domega = phase._ringdown_ansatz_domega(
            time, phase_coeffs.c1, phase_coeffs.c2, phase_coeffs.c3, phase_coeffs.c4
        )
        return phase_coeffs.omegaRING + phase_coeffs.c1 * domega

    else:
        # Intermediate
        # omega = omegaRING * (1 - w)
        # w = 1 - omegaPeak/omegaRING + x * (domegaPeak/alpha1RD + x * (c1 + x * (c2 + x * c3)))
        # x = arcsinh(alpha1RD * t)

        x = jnp.arcsinh(phase_coeffs.alpha1RD * time)
        x2 = x * x
        x3 = x * x2
        x4 = x * x3

        w = (
            1.0
            - phase_coeffs.omegaPeak / phase_coeffs.omegaRING
            + x
            * (
                phase_coeffs.domegaPeak / phase_coeffs.alpha1RD
                + x
                * (
                    phase_coeffs.omegaMergerC1
                    + x * (phase_coeffs.omegaMergerC2 + x * phase_coeffs.omegaMergerC3)
                )
            )
        )

        return phase_coeffs.omegaRING * (1.0 - w)


def _solve_inspiral_amplitude_system(
    times: jnp.ndarray,
    amp_vals: jnp.ndarray,
    eta: float,
    pn_real: jnp.ndarray,
    pn_imag: jnp.ndarray,
    phase_coeffs: phase.PhaseCoeffs22,
) -> tuple:
    """Solve for inspiral pseudo-PN coefficients."""
    fac0 = 2.0 * eta * jnp.sqrt(16.0 * jnp.pi / 5.0)

    # We need omega at collocation points
    # Since times are array, we map _imr_omega
    # But _imr_omega takes scalar time.
    # We can use vmap or list comprehension.
    # Since it's small (3 points), list comprehension is fine.

    omegas = jnp.array([_imr_omega(t, phase_coeffs, eta) for t in times])

    xx = jnp.power(0.5 * omegas, 2.0 / 3.0)
    xxhalf = jnp.sqrt(xx)
    xx4 = xx * xx * xx * xx

    # Compute PN offset
    # We pass pseudo_pn = [0,0,0] to get just PN part
    pseudo_pn_zero = jnp.zeros(3)
    amp_offsets = jnp.array(
        [
            jnp.real(
                _inspiral_ansatz_amplitude(x, fac0, pn_real, pn_imag, pseudo_pn_zero)
            )
            for x in xx
        ]
    )

    B = (1.0 / fac0 / xx) * (amp_vals - amp_offsets)

    # Matrix
    # c1 x^4 + c2 x^4.5 + c3 x^5
    # But we divided B by (fac0 * x), so we are solving for:
    # c1 x^3 + c2 x^3.5 + c3 x^4 ?
    # No, let's check phenomxpy.
    # B[idx] = (1 / self.fac0 / xx) * (self.inspiral_collocation_points[idx, 1] - ampoffset)
    # matrix[idx, jdx] = xx_power (starts at xx4)
    # Wait, if we divide by xx, shouldn't matrix powers be reduced?
    # In phenomxpy:
    # ampoffset = np.real(numba_inspiral_ansatz_amplitude(xx, ..., pseudo_pn_coeffs))
    # Wait, numba_inspiral_ansatz_amplitude includes pseudo_pn terms if passed.
    # But here we pass pseudo_pn_coeffs which are 0 initially?
    # Yes.
    # So ampoffset is just PN part.
    # The equation is: Amp = PN + fac0 * x * (c1 x^4 + c2 x^4.5 + c3 x^5)
    # Amp - PN = fac0 * x * (c1 x^4 + ...)
    # (Amp - PN) / (fac0 * x) = c1 x^4 + c2 x^4.5 + c3 x^5

    # So matrix powers are x^4, x^4.5, x^5.

    matrix = jnp.zeros((3, 3))
    for i in range(3):
        x_val = xx[i]
        x_val_half = xxhalf[i]
        x_val_4 = xx4[i]

        matrix = matrix.at[i, 0].set(x_val_4)
        matrix = matrix.at[i, 1].set(x_val_4 * x_val_half)
        matrix = matrix.at[i, 2].set(x_val_4 * x_val)

    solution = jnp.linalg.solve(matrix, B)

    return solution[0], solution[1], solution[2]


def _der_complex_amp_orientation(
    time: float,
    eta: float,
    pn_real: jnp.ndarray,
    pn_imag: jnp.ndarray,
    pseudo_pn: jnp.ndarray,
    phase_coeffs: phase.PhaseCoeffs22,
    return_phase: bool = False,
) -> float | Array:
    """Compute derivative of complex inspiral amplitude."""
    omega = _imr_omega(time, phase_coeffs, eta)
    x = jnp.power(omega * 0.5, 2.0 / 3.0)

    xhalf = jnp.sqrt(x)
    x1half = x * xhalf
    x2 = x * x
    x2half = x2 * xhalf
    x3 = x2 * x
    x3half = x3 * xhalf
    x4 = x2 * x2
    x4half = x4 * xhalf
    x5 = x3 * x2

    # Real part
    ampreal = (
        pn_real[0]
        + pn_real[1] * xhalf
        + pn_real[2] * x
        + pn_real[3] * x1half
        + pn_real[4] * x2
        + pn_real[5] * x2half
        + pn_real[6] * x3
        + pn_real[7] * x3half
        + pn_real[8] * jnp.log(16.0 * x) * x3
        + pseudo_pn[0] * x4
        + pseudo_pn[1] * x4half
        + pseudo_pn[2] * x5
    )

    # Imaginary part
    ampimag = (
        pn_imag[0] * xhalf
        + pn_imag[1] * x
        + pn_imag[2] * x1half
        + pn_imag[3] * x2
        + pn_imag[4] * x2half
        + pn_imag[5] * x3
        + pn_imag[6] * x3half
    )

    # Derivatives w.r.t x
    dampreal = (
        0.5 * pn_real[1] / xhalf
        + pn_real[2]
        + 1.5 * pn_real[3] * xhalf
        + 2.0 * pn_real[4] * x
        + 2.5 * pn_real[5] * x1half
        + 3.0 * pn_real[6] * x2
        + 3.5 * pn_real[7] * x2half
        + pn_real[8] * x2 * (1.0 + 3.0 * jnp.log(16.0 * x))
        + 4.0 * pseudo_pn[0] * x3
        + 4.5 * pseudo_pn[1] * x3half
        + 5.0 * pseudo_pn[2] * x4
    )

    dampimag = (
        0.5 * pn_imag[0] / xhalf
        + pn_imag[1]
        + 1.5 * pn_imag[2] * xhalf
        + 2.0 * pn_imag[3] * x
        + 2.5 * pn_imag[4] * x1half
        + 3.0 * pn_imag[5] * x2
        + 3.5 * pn_imag[6] * x2half
    )

    der_x_per_omega = jnp.cbrt(2.0 / omega) / 3.0

    # domega/dt
    if time < phase_coeffs.inspiral_cut:
        omega_pn_coeffs = jnp.array(
            [
                phase_coeffs.omega1PN,
                phase_coeffs.omega1halfPN,
                phase_coeffs.omega2PN,
                phase_coeffs.omega2halfPN,
                phase_coeffs.omega3PN,
                phase_coeffs.omega3halfPN,
            ]
        )
        omega_pseudo_pn_coeffs = jnp.array(
            [
                phase_coeffs.omegaInspC1,
                phase_coeffs.omegaInspC2,
                phase_coeffs.omegaInspC3,
                phase_coeffs.omegaInspC4,
                phase_coeffs.omegaInspC5,
                phase_coeffs.omegaInspC6,
            ]
        )
        der_omega_per_t = phase._inspiral_ansatz_domega(
            time, eta, omega_pn_coeffs, omega_pseudo_pn_coeffs
        )
    else:
        # Intermediate derivative
        ascut = jnp.arcsinh(phase_coeffs.alpha1RD * time)
        der_omega_per_t = (
            -phase_coeffs.omegaRING
            / jnp.sqrt(1.0 + (phase_coeffs.alpha1RD * time) ** 2)
            * (
                phase_coeffs.domegaPeak
                + phase_coeffs.alpha1RD
                * (
                    2.0 * phase_coeffs.omegaMergerC1 * ascut
                    + 3.0 * phase_coeffs.omegaMergerC2 * ascut * ascut
                    + 4.0 * phase_coeffs.omegaMergerC3 * ascut**3
                )
            )
        )

    amp = jnp.abs(ampreal + 1j * ampimag)
    fac0 = 2.0 * eta * jnp.sqrt(16.0 * jnp.pi / 5.0)

    if return_phase:
        return (
            (dampimag * ampreal - dampreal * ampimag)
            / (amp * amp)
            * der_x_per_omega
            * der_omega_per_t
        )
    else:
        return (
            fac0
            * (ampreal * (dampreal * x + ampreal) + ampimag * (dampimag * x + ampimag))
            / amp
            * der_x_per_omega
            * der_omega_per_t
        )


def _solve_intermediate_amplitude_system(
    tCut: float,
    tcpMerger: float,
    tshift: float,
    alpha1RD: float,
    ampPeak: float,
    ampMergerCP1: float,
    eta: float,
    pn_real: jnp.ndarray,
    pn_imag: jnp.ndarray,
    pseudo_pn: jnp.ndarray,
    phase_coeffs: phase.PhaseCoeffs22,
) -> tuple[Array, Array, Array, Array, Array]:
    """Solve for intermediate amplitude coefficients."""
    fac0 = 2.0 * eta * jnp.sqrt(16.0 * jnp.pi / 5.0)

    # Compute omega at tCut
    omega_cut = _imr_omega(tCut, phase_coeffs, eta)
    x_cut = jnp.power(omega_cut * 0.5, 2.0 / 3.0)

    # Inspiral amplitude at tCut
    ampinsp_cplx = _inspiral_ansatz_amplitude(x_cut, fac0, pn_real, pn_imag, pseudo_pn)
    ampinsp = ampinsp_cplx

    # Match sign
    ampinsp = jnp.copysign(jnp.abs(ampinsp), jnp.real(ampinsp))

    phi = alpha1RD * (tCut - tshift)
    phi2 = 2.0 * phi

    sech1 = 1.0 / jnp.cosh(phi)
    sech2 = 1.0 / jnp.cosh(phi2)

    matrix = jnp.zeros((4, 4))
    B = jnp.zeros(4)

    # Row 0: Match amplitude at tCut
    matrix = matrix.at[0, 0].set(1.0)
    matrix = matrix.at[0, 1].set(sech1)
    matrix = matrix.at[0, 2].set(jnp.power(sech2, 1.0 / 7.0))
    matrix = matrix.at[0, 3].set((tCut - tshift) ** 2)
    B = B.at[0].set(ampinsp)

    # Row 1: Match amplitude at tcpMerger
    phib = alpha1RD * (tcpMerger - tshift)
    sech1b = 1.0 / jnp.cosh(phib)
    sech2b = 1.0 / jnp.cosh(2.0 * phib)

    matrix = matrix.at[1, 0].set(1.0)
    matrix = matrix.at[1, 1].set(sech1b)
    matrix = matrix.at[1, 2].set(jnp.power(sech2b, 1.0 / 7.0))
    matrix = matrix.at[1, 3].set((tcpMerger - tshift) ** 2)
    B = B.at[1].set(ampMergerCP1)

    # Row 2: Match amplitude at peak (t=0)
    # t=0 means phi = alpha1RD * (-tshift)
    # Wait, phenomxpy says:
    # matrix[2, 0] = 1
    # matrix[2, 1] = 1
    # matrix[2, 2] = 1
    # matrix[2, 3] = 0
    # B[2] = self.ampPeak
    # This implies t=tshift?
    # Yes, tshift is the peak time for amplitude.
    # So at t=tshift, phi=0, sech=1.

    matrix = matrix.at[2, 0].set(1.0)
    matrix = matrix.at[2, 1].set(1.0)
    matrix = matrix.at[2, 2].set(1.0)
    matrix = matrix.at[2, 3].set(0.0)
    B = B.at[2].set(ampPeak)

    # Row 3: Match derivative at tCut
    amp2 = ampinsp_cplx
    dampMECO = jnp.copysign(1.0, jnp.real(amp2)) * _der_complex_amp_orientation(
        tCut, eta, pn_real, pn_imag, pseudo_pn, phase_coeffs, return_phase=False
    )

    tanh = jnp.tanh(phi)
    sinh = jnp.sinh(phi2)

    aux1 = -alpha1RD * sech1 * tanh
    aux2 = (
        (-2.0 / 7.0) * alpha1RD * sinh * jnp.power(sech2, 8.0 / 7.0)
    )  # Wait, sinh * sech2^(8/7)?
    # phenomxpy: aux2 = (-2 / 7) * self.alpha1RD * sinh * np.power(sech2, 8 / 7)
    # Yes.
    aux3 = 2.0 * (tCut - tshift)

    matrix = matrix.at[3, 0].set(0.0)
    matrix = matrix.at[3, 1].set(aux1)
    matrix = matrix.at[3, 2].set(aux2)
    matrix = matrix.at[3, 3].set(aux3)
    B = B.at[3].set(dampMECO)

    solution = jnp.linalg.solve(matrix, B)

    return solution[0], solution[1], solution[2], solution[3], dampMECO


# def _intermediate_ansatz_amplitude(
#     time: float,
#     c1: float,
#     c2: float,
#     c3: float,
#     c4: float,
#     alpha: float,
#     tshift: float,
# ) -> float | Array:
#     """
#     Evaluate intermediate amplitude ansatz.

#     Amp(t) = c1 + c2 * sech(alpha * dt) + c3 * sech(2 * alpha * dt)^(1/7) + c4 * dt^2
#     """
#     dt = time - tshift
#     phi = alpha * dt
#     phi2 = 2.0 * phi

#     sech1 = 1.0 / jnp.cosh(phi)
#     sech2 = 1.0 / jnp.cosh(phi2)

#     return c1 + c2 * sech1 + c3 * jnp.power(sech2, 1.0 / 7.0) + c4 * dt * dt


# def _ringdown_ansatz_amplitude(
#     time: float,
#     c1: float,
#     c2: float,
#     c3: float,
#     c4: float,
#     tshift: float,
# ) -> float | Array:
#     """
#     Evaluate ringdown amplitude ansatz.

#     Amp(t) = c1 * tanh(c2 * (t - tshift) + c3) + c4
#     """
#     dt = time - tshift
#     return c1 * jnp.tanh(c2 * dt + c3) + c4
