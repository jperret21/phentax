# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
Tests for amplitude coefficients comparison with phenomxpy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Enable float64 for precision
jax.config.update("jax_enable_x64", True)

from phentax import amplitude, phase

# Try to import phenomxpy for comparison
try:
    import phenomxpy
    from phenomxpy.phenomt.phenomt import IMRPhenomT

    PHENOMXPY_AVAILABLE = True
except ImportError:
    PHENOMXPY_AVAILABLE = False


def assert_close(actual, expected, rtol=1e-10, atol=1e-12, name=""):
    """Helper to assert closeness with detailed error message."""
    # Handle JAX arrays
    if hasattr(actual, "item"):
        actual = actual.item()
    if hasattr(expected, "item"):
        expected = expected.item()

    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as e:
        raise AssertionError(
            f"{name} mismatch:\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {actual}\n"
            f"  Abs diff: {abs(actual - expected)}\n"
            f"  Rel diff: {abs(actual - expected) / abs(expected) if expected != 0 else 'inf'}"
        ) from e


@pytest.mark.skipif(not PHENOMXPY_AVAILABLE, reason="phenomxpy not installed")
class TestAmplitudeCoeffs22:
    """
    Test suite for AmplitudeCoeffs22 against phenomxpy.
    """

    @pytest.mark.parametrize(
        "eta,chi1,chi2",
        [
            (0.25, 0.0, 0.0),  # Schwarzschild
            (0.25, 0.5, 0.5),  # Equal spin aligned
            (0.24, 0.3, -0.1),  # Unequal mass, unequal spin
            (0.2, 0.6, 0.2),  # More unequal mass
            (0.1, -0.5, 0.4),  # Low mass ratio
        ],
    )
    def test_pn_coefficients(self, eta, chi1, chi2):
        """Test PN coefficients."""
        # Compute phentax coefficients
        phase_coeffs = phase.compute_phase_coeffs_22(eta, chi1, chi2)
        coeffs = amplitude.compute_amplitude_coeffs_22(eta, chi1, chi2, phase_coeffs)

        # Compute phenomxpy coefficients
        p = IMRPhenomT(eta=eta, s1=chi1, s2=chi2, f_min=20.0, total_mass=100)
        pamp = p.pAmp

        # Check PN coefficients
        assert_close(coeffs.ampN, pamp.ampN, name="ampN")
        assert_close(coeffs.amp0halfPNreal, pamp.amp0halfPNreal, name="amp0halfPNreal")
        assert_close(coeffs.amp1PNreal, pamp.amp1PNreal, name="amp1PNreal")
        assert_close(coeffs.amp1halfPNreal, pamp.amp1halfPNreal, name="amp1halfPNreal")
        assert_close(coeffs.amp2PNreal, pamp.amp2PNreal, name="amp2PNreal")
        assert_close(coeffs.amp2halfPNreal, pamp.amp2halfPNreal, name="amp2halfPNreal")
        assert_close(coeffs.amp3PNreal, pamp.amp3PNreal, name="amp3PNreal")
        assert_close(coeffs.amp3halfPNreal, pamp.amp3halfPNreal, name="amp3halfPNreal")
        assert_close(coeffs.amplog, pamp.amplog, name="amplog")

        assert_close(coeffs.amp0halfPNimag, pamp.amp0halfPNimag, name="amp0halfPNimag")
        assert_close(coeffs.amp1PNimag, pamp.amp1PNimag, name="amp1PNimag")
        assert_close(coeffs.amp1halfPNimag, pamp.amp1halfPNimag, name="amp1halfPNimag")
        assert_close(coeffs.amp2PNimag, pamp.amp2PNimag, name="amp2PNimag")
        assert_close(coeffs.amp2halfPNimag, pamp.amp2halfPNimag, name="amp2halfPNimag")
        assert_close(coeffs.amp3PNimag, pamp.amp3PNimag, name="amp3PNimag")
        assert_close(coeffs.amp3halfPNimag, pamp.amp3halfPNimag, name="amp3halfPNimag")

    @pytest.mark.parametrize(
        "eta,chi1,chi2",
        [
            (0.25, 0.0, 0.0),
            (0.25, 0.5, 0.5),
            (0.24, 0.3, -0.1),
            (0.2, 0.6, 0.2),
            (0.1, -0.5, 0.4),
        ],
    )
    def test_inspiral_coefficients(self, eta, chi1, chi2):
        """Test inspiral pseudo-PN coefficients."""
        phase_coeffs = phase.compute_phase_coeffs_22(eta, chi1, chi2)
        coeffs = amplitude.compute_amplitude_coeffs_22(eta, chi1, chi2, phase_coeffs)

        p = IMRPhenomT(eta=eta, s1=chi1, s2=chi2, f_min=20.0, total_mass=100)
        pamp = p.pAmp

        # Check collocation points values
        assert_close(coeffs.ampInspCP1, pamp.ampInspCP1, name="ampInspCP1")
        assert_close(coeffs.ampInspCP2, pamp.ampInspCP2, name="ampInspCP2")
        assert_close(coeffs.ampInspCP3, pamp.ampInspCP3, name="ampInspCP3")

        # Check coefficients
        assert_close(coeffs.inspC1, pamp.inspC1, name="inspC1")
        assert_close(coeffs.inspC2, pamp.inspC2, name="inspC2")
        assert_close(coeffs.inspC3, pamp.inspC3, name="inspC3")

    @pytest.mark.parametrize(
        "eta,chi1,chi2",
        [
            (0.25, 0.0, 0.0),
            (0.25, 0.5, 0.5),
            (0.24, 0.3, -0.1),
            (0.2, 0.6, 0.2),
            (0.1, -0.5, 0.4),
        ],
    )
    def test_ringdown_coefficients(self, eta, chi1, chi2):
        """Test ringdown coefficients."""
        phase_coeffs = phase.compute_phase_coeffs_22(eta, chi1, chi2)
        coeffs = amplitude.compute_amplitude_coeffs_22(eta, chi1, chi2, phase_coeffs)

        p = IMRPhenomT(eta=eta, s1=chi1, s2=chi2, f_min=20.0, total_mass=100)
        pamp = p.pAmp

        assert_close(coeffs.alpha1RD, pamp.alpha1RD, name="alpha1RD")
        assert_close(coeffs.alpha2RD, pamp.alpha2RD, name="alpha2RD")
        assert_close(coeffs.alpha21RD, pamp.alpha21RD, name="alpha21RD")

        assert_close(coeffs.ampPeak, pamp.ampPeak, name="ampPeak")
        assert_close(coeffs.c1, pamp.c1, name="c1")
        assert_close(coeffs.c2, pamp.c2, name="c2")
        assert_close(coeffs.c3, pamp.c3, name="c3")
        assert_close(coeffs.c4, pamp.c4, name="c4")

    @pytest.mark.parametrize(
        "eta,chi1,chi2",
        [
            (0.25, 0.0, 0.0),
            (0.25, 0.5, 0.5),
            (0.24, 0.3, -0.1),
            (0.2, 0.6, 0.2),
            (0.1, -0.5, 0.4),
        ],
    )
    def test_intermediate_coefficients(self, eta, chi1, chi2):
        """Test intermediate coefficients."""
        phase_coeffs = phase.compute_phase_coeffs_22(eta, chi1, chi2)
        coeffs = amplitude.compute_amplitude_coeffs_22(eta, chi1, chi2, phase_coeffs)

        p = IMRPhenomT(eta=eta, s1=chi1, s2=chi2, f_min=20.0, total_mass=100)
        pamp = p.pAmp

        assert_close(coeffs.tshift, pamp.tshift, name="tshift")
        assert_close(coeffs.ampMergerCP1, pamp.ampMergerCP1, name="ampMergerCP1")
        assert_close(coeffs.dampMECO, pamp.dampMECO, name="dampMECO")

        assert_close(coeffs.mergerC1, pamp.mergerC1, name="mergerC1")
        assert_close(coeffs.mergerC2, pamp.mergerC2, name="mergerC2")
        assert_close(coeffs.mergerC3, pamp.mergerC3, name="mergerC3")
        assert_close(coeffs.mergerC4, pamp.mergerC4, name="mergerC4")
