# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
Utility functions for phentax.

Contains helper functions for mass ratios, spins, unit conversions,
and spin-weighted spherical harmonics, all implemented in pure JAX.
"""

from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
from jax import lax

from .constants import C_SI, MPC_TO_M, MTSUN_SI

# =============================================================================
# Mass ratio utilities
# =============================================================================


@jax.jit
def m1ofeta(
    eta: Union[float, jnp.ndarray], total_mass: Union[float, jnp.ndarray] = 1.0
) -> Union[float, jnp.ndarray]:
    """
    Compute primary mass from symmetric mass ratio.

    Parameters
    ----------
    eta : float
        Symmetric mass ratio in [0, 0.25].
    total_mass : float, default 1.0
        Total mass (m1 + m2). If 1.0, returns dimensionless mass fraction.

    Returns
    -------
    float
        Primary (larger) mass.
    """
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    return 0.5 * total_mass * (1.0 + delta)


@jax.jit
def m2ofeta(
    eta: Union[float, jnp.ndarray], total_mass: Union[float, jnp.ndarray] = 1.0
) -> Union[float, jnp.ndarray]:
    """
    Compute secondary mass from symmetric mass ratio.

    Parameters
    ----------
    eta : Union[float, jnp.ndarray]
        Symmetric mass ratio in [0, 0.25].
    total_mass : Union[float, jnp.ndarray], default 1.0
        Total mass (m1 + m2). If 1.0, returns dimensionless mass fraction.

    Returns
    -------
    Union[float, jnp.ndarray]
        Secondary (smaller) mass.
    """
    delta = jnp.sqrt(1.0 - 4.0 * eta)
    return 0.5 * total_mass * (1.0 - delta)


@jax.jit
def qofeta(eta: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """
    Compute mass ratio q = m1/m2 >= 1 from symmetric mass ratio.

    Parameters
    ----------
    eta : Union[float, jnp.ndarray]
        Symmetric mass ratio in [0, 0.25].

    Returns
    -------
    Union[float, jnp.ndarray]
        Mass ratio q >= 1.
    """
    m1 = m1ofeta(eta)
    m2 = m2ofeta(eta)
    return m1 / m2


@jax.jit
def eta_from_q(q: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """
    Compute symmetric mass ratio from mass ratio q = m1/m2 >= 1.

    Parameters
    ----------
    q : Union[float, jnp.ndarray]
        Mass ratio q >= 1.

    Returns
    -------
    Union[float, jnp.ndarray]
        Symmetric mass ratio eta.
    """
    return q / (1.0 + q) ** 2


# =============================================================================
# Spin utilities
# =============================================================================


@jax.jit
def chi_eff(
    eta: Union[float, jnp.ndarray],
    s1z: Union[float, jnp.ndarray],
    s2z: Union[float, jnp.ndarray],
) -> Union[float, jnp.ndarray]:
    """
    Compute effective spin parameter.

    Parameters
    ----------
    eta : Union[float, jnp.ndarray]
        Symmetric mass ratio.
    s1z : Union[float, jnp.ndarray]
        Dimensionless spin of primary along orbital angular momentum.
    s2z : Union[float, jnp.ndarray]
        Dimensionless spin of secondary along orbital angular momentum.

    Returns
    -------
    Union[float, jnp.ndarray]
        Effective spin chi_eff = (m1*s1z + m2*s2z) / (m1 + m2).
    """
    m1 = m1ofeta(eta)
    m2 = m2ofeta(eta)
    return (m1 * s1z + m2 * s2z) / (m1 + m2)


@jax.jit
def sTotR(
    eta: Union[float, jnp.ndarray],
    s1z: Union[float, jnp.ndarray],
    s2z: Union[float, jnp.ndarray],
) -> Union[float, jnp.ndarray]:
    """
    Compute reduced total spin parameter S = (m1^2*s1z + m2^2*s2z) / (m1^2 + m2^2).

    This is the spin combination used in PhenomT fits.

    Parameters
    ----------
    eta : Union[float, jnp.ndarray]
        Symmetric mass ratio.
    s1z : Union[float, jnp.ndarray]
        Dimensionless spin of primary along orbital angular momentum.
    s2z : Union[float, jnp.ndarray]
        Dimensionless spin of secondary along orbital angular momentum.

    Returns
    -------
    Union[float, jnp.ndarray]
        Reduced total spin S.
    """
    m1 = m1ofeta(eta)
    m2 = m2ofeta(eta)
    m1_sq = m1 * m1
    m2_sq = m2 * m2
    return (m1_sq * s1z + m2_sq * s2z) / (m1_sq + m2_sq)


# =============================================================================
# Unit conversions
# =============================================================================


@jax.jit
def hz_to_mf(
    f_hz: Union[float, jnp.ndarray], total_mass: Union[float, jnp.ndarray]
) -> Union[float, jnp.ndarray]:
    """
    Convert frequency from Hz to dimensionless units (Mf).

    Parameters
    ----------
    f_hz : Union[float, jnp.ndarray]
        Frequency in Hz.
    total_mass : Union[float, jnp.ndarray]
        Total mass in solar masses.

    Returns
    -------
    Union[float, jnp.ndarray]
        Dimensionless frequency Mf.
    """
    return f_hz * total_mass * MTSUN_SI


@jax.jit
def mf_to_hz(
    mf: Union[float, jnp.ndarray], total_mass: Union[float, jnp.ndarray]
) -> Union[float, jnp.ndarray]:
    """
    Convert frequency from dimensionless units (Mf) to Hz.

    Parameters
    ----------
    mf : Union[float, jnp.ndarray]
        Dimensionless frequency Mf.
    total_mass : Union[float, jnp.ndarray]
        Total mass in solar masses.

    Returns
    -------
    Union[float, jnp.ndarray]
        Frequency in Hz.
    """
    return mf / (total_mass * MTSUN_SI)


@jax.jit
def second_to_mass(
    t_sec: Union[float, jnp.ndarray], total_mass: Union[float, jnp.ndarray]
) -> Union[float, jnp.ndarray]:
    """
    Convert time from seconds to dimensionless units (t/M).

    Parameters
    ----------
    t_sec : Union[float, jnp.ndarray]
        Time in seconds.
    total_mass : Union[float, jnp.ndarray]
        Total mass in solar masses.

    Returns
    -------
    Union[float, jnp.ndarray]
        Dimensionless time t/M.
    """
    return t_sec / (total_mass * MTSUN_SI)


@jax.jit
def mass_to_second(
    t_m: Union[float, jnp.ndarray], total_mass: Union[float, jnp.ndarray]
) -> Union[float, jnp.ndarray]:
    """
    Convert time from dimensionless units (t/M) to seconds.

    Parameters
    ----------
    t_m : Union[float, jnp.ndarray]
        Dimensionless time t/M.
    total_mass : Union[float, jnp.ndarray]
        Total mass in solar masses.

    Returns
    -------
    Union[float, jnp.ndarray]
        Time in seconds.
    """
    return t_m * total_mass * MTSUN_SI


@jax.jit
def amp_nrto_si(
    h_nr: jnp.ndarray,
    distance_mpc: Union[float, jnp.ndarray],
    total_mass: Union[float, jnp.ndarray],
) -> jnp.ndarray:
    """
    Convert strain amplitude from NR units to SI units.

    Parameters
    ----------
    h_nr : array
        Strain in NR units (dimensionless, at unit distance).
    distance_mpc : Union[float, jnp.ndarray]
        Luminosity distance in Megaparsecs.
    total_mass : Union[float, jnp.ndarray]
        Total mass in solar masses.

    Returns
    -------
    array
        Strain in SI units.
    """
    # h_SI = h_NR * (G * M / c^2) / D = h_NR * M_in_meters / D_in_meters
    mass_in_seconds = total_mass * MTSUN_SI
    mass_in_meters = mass_in_seconds * C_SI
    distance_in_meters = distance_mpc * MPC_TO_M
    return h_nr * mass_in_meters / distance_in_meters


# =============================================================================
# Mode utilities
# =============================================================================


def mode_to_int(ell: int, emm: int) -> int:
    """
    Convert (ell, m) mode indices to integer key.

    Parameters
    ----------
    ell : int
        Orbital angular momentum quantum number.
    emm : int
        Azimuthal quantum number.

    Returns
    -------
    int
        Integer key: 10*ell + |m| for positive m modes.
    """
    return 10 * ell + abs(emm)


# =============================================================================
# Spin-weighted spherical harmonics
# =============================================================================


@partial(jax.jit, static_argnums=(2, 3, 4))
def spin_weighted_spherical_harmonic(
    theta: float,
    phi: float,
    s: int = -2,
    ell: int = 2,
    emm: int = 2,
) -> jnp.ndarray:
    """
    Compute spin-weighted spherical harmonic Y_{s,ell,m}(theta, phi).

    Uses explicit formulas for s=-2 harmonics needed for gravitational waves.

    Parameters
    ----------
    theta : float
        Polar angle (inclination) in radians.
    phi : float
        Azimuthal angle in radians.
    s : int, default -2
        Spin weight (must be -2 for GW).
    ell : int, default 2
        Orbital angular momentum quantum number.
    emm : int, default 2
        Azimuthal quantum number.

    Returns
    -------
    jnp.ndarray
        Value of the spin-weighted spherical harmonic.
    """
    # For gravitational waves we only need s=-2
    # Using explicit formulas from LAL

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    cos_half = jnp.cos(theta / 2)
    sin_half = jnp.sin(theta / 2)

    # Compute the angular part based on (ell, m)
    # All functions return float for consistency with lax.switch
    def ylm_22():
        return jnp.sqrt(5.0 / (64.0 * jnp.pi)) * (1 + cos_theta) ** 2

    def ylm_2m2():
        return jnp.sqrt(5.0 / (64.0 * jnp.pi)) * (1 - cos_theta) ** 2

    def ylm_21():
        return jnp.sqrt(5.0 / (16.0 * jnp.pi)) * sin_theta * (1 + cos_theta)

    def ylm_2m1():
        return jnp.sqrt(5.0 / (16.0 * jnp.pi)) * sin_theta * (1 - cos_theta)

    def ylm_20():
        return jnp.sqrt(15.0 / (32.0 * jnp.pi)) * sin_theta**2

    def ylm_33():
        return -jnp.sqrt(21.0 / (128.0 * jnp.pi)) * (1 + cos_theta) ** 2 * sin_theta

    def ylm_3m3():
        return jnp.sqrt(21.0 / (128.0 * jnp.pi)) * (1 - cos_theta) ** 2 * sin_theta

    def ylm_32():
        return (
            jnp.sqrt(7.0 / (64.0 * jnp.pi))
            * (2 + 3 * cos_theta)
            * (1 + cos_theta)
            * sin_theta
        )

    def ylm_3m2():
        return (
            jnp.sqrt(7.0 / (64.0 * jnp.pi))
            * (2 - 3 * cos_theta)
            * (1 - cos_theta)
            * sin_theta
        )

    def ylm_31():
        return (
            jnp.sqrt(35.0 / (128.0 * jnp.pi))
            * (1 + cos_theta)
            * (1 - 3 * cos_theta + 4 * cos_theta**2)
            * sin_theta
            / (1 + cos_theta + 1e-30)
        )

    def ylm_3m1():
        return (
            -jnp.sqrt(35.0 / (128.0 * jnp.pi))
            * (1 - cos_theta)
            * (1 + 3 * cos_theta + 4 * cos_theta**2)
            * sin_theta
            / (1 - cos_theta + 1e-30)
        )

    def ylm_44():
        return (
            3 * jnp.sqrt(7.0 / (128.0 * jnp.pi)) * (1 + cos_theta) ** 2 * sin_theta**2
        )

    def ylm_4m4():
        return (
            3 * jnp.sqrt(7.0 / (128.0 * jnp.pi)) * (1 - cos_theta) ** 2 * sin_theta**2
        )

    def ylm_55():
        return (
            -jnp.sqrt(330.0 / (1024.0 * jnp.pi)) * (1 + cos_theta) ** 2 * sin_theta**3
        )

    def ylm_5m5():
        return jnp.sqrt(330.0 / (1024.0 * jnp.pi)) * (1 - cos_theta) ** 2 * sin_theta**3

    def ylm_default():
        return 0.0  # Return float, same as other branches

    # Use Wigner d-matrix based formula for s=-2 spherical harmonics
    # For now, implement explicit cases for modes we support
    mode_key = 10 * ell + emm

    # Select the appropriate angular function
    angular = lax.switch(
        mode_key + 55,  # Offset to make all indices non-negative
        [
            ylm_5m5,  # -55 + 55 = 0
            *[ylm_default] * 10,
            ylm_4m4,  # -44 + 55 = 11
            *[ylm_default] * 10,
            ylm_3m3,  # -33 + 55 = 22
            ylm_3m2,  # -32 + 55 = 23
            ylm_3m1,  # -31 + 55 = 24
            *[ylm_default] * 6,
            ylm_2m2,  # -22 + 55 = 33
            ylm_2m1,  # -21 + 55 = 34
            ylm_20,  # -20 + 55 = 35 (20 mode)
            *[ylm_default] * 40,
            ylm_20,  # 20 + 55 = 75
            ylm_21,  # 21 + 55 = 76
            ylm_22,  # 22 + 55 = 77
            *[ylm_default] * 8,
            ylm_31,  # 31 + 55 = 86
            ylm_32,  # 32 + 55 = 87
            ylm_33,  # 33 + 55 = 88
            *[ylm_default] * 10,
            ylm_44,  # 44 + 55 = 99
            *[ylm_default] * 10,
            ylm_55,  # 55 + 55 = 110
        ],
    )

    # Add the azimuthal phase
    return angular * jnp.exp(1j * emm * phi)


# Simpler implementation using conditionals (fallback)
@jax.jit
def _swsh_22(theta: float, phi: float) -> jnp.ndarray:
    """Spin-weighted spherical harmonic Y_{-2,2,2}."""
    cos_theta = jnp.cos(theta)
    angular = jnp.sqrt(5.0 / (64.0 * jnp.pi)) * (1 + cos_theta) ** 2
    return angular * jnp.exp(2j * phi)


@jax.jit
def _swsh_2m2(theta: float, phi: float) -> jnp.ndarray:
    """Spin-weighted spherical harmonic Y_{-2,2,-2}."""
    cos_theta = jnp.cos(theta)
    angular = jnp.sqrt(5.0 / (64.0 * jnp.pi)) * (1 - cos_theta) ** 2
    return angular * jnp.exp(-2j * phi)


@jax.jit
def _swsh_21(theta: float, phi: float) -> jnp.ndarray:
    """Spin-weighted spherical harmonic Y_{-2,2,1}."""
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    angular = jnp.sqrt(5.0 / (16.0 * jnp.pi)) * sin_theta * (1 + cos_theta)
    return angular * jnp.exp(1j * phi)


@jax.jit
def _swsh_2m1(theta: float, phi: float) -> jnp.ndarray:
    """Spin-weighted spherical harmonic Y_{-2,2,-1}."""
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    angular = jnp.sqrt(5.0 / (16.0 * jnp.pi)) * sin_theta * (1 - cos_theta)
    return angular * jnp.exp(-1j * phi)


@jax.jit
def _swsh_20(theta: float, phi: float) -> jnp.ndarray:
    """Spin-weighted spherical harmonic Y_{-2,2,0}."""
    sin_theta = jnp.sin(theta)
    angular = jnp.sqrt(15.0 / (32.0 * jnp.pi)) * sin_theta**2
    return angular  # exp(0) = 1


@jax.jit
def _swsh_33(theta: float, phi: float) -> jnp.ndarray:
    """Spin-weighted spherical harmonic Y_{-2,3,3}."""
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    angular = -jnp.sqrt(21.0 / (128.0 * jnp.pi)) * (1 + cos_theta) ** 2 * sin_theta
    return angular * jnp.exp(3j * phi)


@jax.jit
def _swsh_3m3(theta: float, phi: float) -> jnp.ndarray:
    """Spin-weighted spherical harmonic Y_{-2,3,-3}."""
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    angular = jnp.sqrt(21.0 / (128.0 * jnp.pi)) * (1 - cos_theta) ** 2 * sin_theta
    return angular * jnp.exp(-3j * phi)


@jax.jit
def _swsh_44(theta: float, phi: float) -> jnp.ndarray:
    """Spin-weighted spherical harmonic Y_{-2,4,4}."""
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    angular = 3 * jnp.sqrt(7.0 / (128.0 * jnp.pi)) * (1 + cos_theta) ** 2 * sin_theta**2
    return angular * jnp.exp(4j * phi)


@jax.jit
def _swsh_4m4(theta: float, phi: float) -> jnp.ndarray:
    """Spin-weighted spherical harmonic Y_{-2,4,-4}."""
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    angular = 3 * jnp.sqrt(7.0 / (128.0 * jnp.pi)) * (1 - cos_theta) ** 2 * sin_theta**2
    return angular * jnp.exp(-4j * phi)


@jax.jit
def _swsh_55(theta: float, phi: float) -> jnp.ndarray:
    """Spin-weighted spherical harmonic Y_{-2,5,5}."""
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    angular = -jnp.sqrt(330.0 / (1024.0 * jnp.pi)) * (1 + cos_theta) ** 2 * sin_theta**3
    return angular * jnp.exp(5j * phi)


@jax.jit
def _swsh_5m5(theta: float, phi: float) -> jnp.ndarray:
    """Spin-weighted spherical harmonic Y_{-2,5,-5}."""
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    angular = jnp.sqrt(330.0 / (1024.0 * jnp.pi)) * (1 - cos_theta) ** 2 * sin_theta**3
    return angular * jnp.exp(-5j * phi)


# Dictionary-style lookup for spherical harmonics
_SWSH_FUNCS = {
    (2, 2): _swsh_22,
    (2, -2): _swsh_2m2,
    (2, 1): _swsh_21,
    (2, -1): _swsh_2m1,
    (2, 0): _swsh_20,
    (3, 3): _swsh_33,
    (3, -3): _swsh_3m3,
    (4, 4): _swsh_44,
    (4, -4): _swsh_4m4,
    (5, 5): _swsh_55,
    (5, -5): _swsh_5m5,
}


def get_swsh(ell: int, emm: int):
    """
    Get the spin-weighted spherical harmonic function for mode (ell, m).

    Parameters
    ----------
    ell : int
        Orbital angular momentum quantum number.
    emm : int
        Azimuthal quantum number.

    Returns
    -------
    Callable
        Function swsh(theta, phi) -> jnp.ndarray.
    """
    key = (ell, emm)
    if key not in _SWSH_FUNCS:
        raise ValueError(f"Mode ({ell}, {emm}) not implemented")
    return _SWSH_FUNCS[key]
