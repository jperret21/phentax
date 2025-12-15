# Copyright (C) 2024 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
phentax: JAX implementation of IMRPhenomT(HM) gravitational waveform model.

This package provides differentiable, JIT-compiled gravitational waveform
generation for compact binary coalescences using JAX.

Main API
--------
compute_polarizations : Compute h_plus and h_cross polarizations
compute_hlm : Compute a single spherical harmonic mode
compute_hlms : Compute multiple spherical harmonic modes
make_params : Create waveform parameters
generate_waveform : Convenience function for complete waveform generation

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

from .config import configure_jax

# Configure JAX for float64 by default
configure_jax()

# Data structures
from .internals import (
    AmpCoeffs,
    DerivedParams,
    ModeCoeffs,
    PhaseCoeffs,
    WaveformParams,
    compute_derived_params,
)

# Core utilities
from .utils import (
    chi_eff,
    eta_from_q,
    hz_to_mf,
    m1ofeta,
    m2ofeta,
    mf_to_hz,
    qofeta,
    spin_weighted_spherical_harmonic,
    sTotR,
)

# Main waveform API
from .waveform import (
    compute_hlm,
    compute_hlms,
    compute_polarizations,
    compute_polarizations_hm,
    generate_waveform,
    make_params,
)

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "configure_jax",
    # Utilities
    "chi_eff",
    "m1ofeta",
    "m2ofeta",
    "qofeta",
    "eta_from_q",
    "sTotR",
    "spin_weighted_spherical_harmonic",
    "hz_to_mf",
    "mf_to_hz",
    # Data structures
    "WaveformParams",
    "DerivedParams",
    "PhaseCoeffs",
    "AmpCoeffs",
    "ModeCoeffs",
    "compute_derived_params",
    # Waveform API
    "make_params",
    "compute_hlm",
    "compute_hlms",
    "compute_polarizations",
    "compute_polarizations_hm",
    "generate_waveform",
]
