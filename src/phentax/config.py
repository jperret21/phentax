# Copyright (C) 2025 Alessandro Santini
# SPDX-License-Identifier: MIT
"""
JAX configuration for phentax.

This module configures JAX for gravitational waveform computations,
enabling float64 precision by default and providing platform controls.
"""

import jax


def configure_jax(enable_x64: bool = True, platform: str | None = None) -> None:
    """
    Configure JAX for phentax.

    Parameters
    ----------
    enable_x64 : bool, default True
        Enable 64-bit floating point precision. Required for waveform accuracy.
    platform : str or None, default None
        Force JAX to use a specific platform ('cpu', 'gpu', 'tpu').
        If None, JAX auto-selects.
    """
    if enable_x64:
        jax.config.update("jax_enable_x64", True)

    if platform is not None:
        jax.config.update("jax_platform_name", platform)
