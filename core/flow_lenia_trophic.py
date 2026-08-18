"""
core/flow_lenia_trophic.py
==========================
Authentic Two-Species Flow-Lenia PDE Trophic Dynamics.

Implements canonical Flow-Lenia physics for two independent continuous density
fields U_prey(x,y,t) and U_pred(x,y,t), coupled via:
  1. Lotka-Volterra cross-species growth modulation (prey suppressed where pred
     is dense; predator energized where prey is dense).
  2. Chemotactic scent fields (Fourier-Gaussian diffusion) driving prey evasion
     and predator hunting via the velocity field.

References:
  - Plantec et al. (2025). Flow-Lenia: Emergent Evolutionary Dynamics.
    Artificial Life 31(2). arXiv:2506.08569.
  - Michel et al. (2025/2026). Exploring Flow-Lenia Universes with a
    Curiosity-driven AI Scientist. arXiv:2505.15998.
  - Moroz (2020). Particle-in-Cell Methods.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple

from core.flow_lenia_jax import (
    compute_sobel_gradients,
    moroz_reintegration_tracking,
)


def build_scent_kernel_fft(H: int, W: int, sigma_px: float = 30.0) -> jnp.ndarray:
    """Precompute rfft2 of a centred Gaussian scent diffusion kernel."""
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    r2 = (yy - H // 2) ** 2 + (xx - W // 2) ** 2
    k = np.exp(-r2 / (2.0 * sigma_px ** 2))
    k = np.roll(k, (-H // 2, -W // 2), axis=(0, 1))
    k /= k.sum()
    return jnp.array(np.fft.rfft2(k))


@jax.jit
def flow_lenia_trophic_step(
    U0: jnp.ndarray,
    U1: jnp.ndarray,
    k_ffts_0: jnp.ndarray,
    k_ffts_1: jnp.ndarray,
    scent_fft: jnp.ndarray,
    mu0: float, sigma0: float, v_scale0: float, alpha0: float,
    chi0: float, lambda0: float, gamma0: float,
    mu1: float, sigma1: float, v_scale1: float, alpha1: float,
    chi1: float, lambda1: float, gamma1: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Advance prey (U0) and predator (U1) by one PDE timestep.

    Prey:
        G0 = mean_k [ 2*exp(-((U0*K0_k - mu0)^2)/(2*sigma0^2)) - 1 ]
        G0_eff = G0 * (1 - lambda0 * U1)      [Lotka-Volterra suppression]
        v0  = tanh(v_scale0 * [(1-alpha0)*grad(G0_eff) - alpha0*grad(U0)]
                   + chi0 * grad(scent_pred))  [flee from predator, chi0 < 0]
        U0' = Moroz(U0, v0)

    Predator:
        G1 = mean_k [ 2*exp(-((U1*K1_k - mu1)^2)/(2*sigma1^2)) - 1 ]
        G1_eff = G1 * (1 + lambda1 * U0)      [Lotka-Volterra energizing]
        v1  = tanh(v_scale1 * [(1-alpha1)*grad(G1_eff) - alpha1*grad(U1)]
                   + chi1 * grad(scent_prey))  [hunt prey, chi1 > 0]
        U1' = Moroz(U1, v1) - gamma1 * U1     [basal starvation]
    """
    H, W = U0.shape

    fft_U0 = jnp.fft.rfft2(U0)
    fft_U1 = jnp.fft.rfft2(U1)

    # Scent fields via Fourier-Gaussian diffusion
    scent0 = jnp.fft.irfft2(fft_U0 * scent_fft, s=(H, W))
    scent1 = jnp.fft.irfft2(fft_U1 * scent_fft, s=(H, W))

    dscent1_x, dscent1_y = compute_sobel_gradients(scent1)  # pred scent grad → prey evasion
    dscent0_x, dscent0_y = compute_sobel_gradients(scent0)  # prey scent grad → pred hunting

    # ── Prey ──────────────────────────────────────────────────────────────────
    U0_conv = jax.vmap(
        lambda k_fft: jnp.fft.irfft2(fft_U0 * k_fft, s=(H, W))
    )(k_ffts_0)

    G0 = jnp.mean(
        2.0 * jnp.exp(-((U0_conv - mu0) ** 2) / (2.0 * sigma0 ** 2 + 1e-9)) - 1.0,
        axis=0,
    )
    G0_eff = G0 * (1.0 - lambda0 * jnp.clip(U1, 0.0, 1.0))

    dG0_x, dG0_y = compute_sobel_gradients(G0_eff)
    dU0_x, dU0_y = compute_sobel_gradients(U0)

    vx0 = jnp.tanh(v_scale0 * ((1.0 - alpha0) * dG0_x - alpha0 * dU0_x) + chi0 * dscent1_x)
    vy0 = jnp.tanh(v_scale0 * ((1.0 - alpha0) * dG0_y - alpha0 * dU0_y) + chi0 * dscent1_y)

    U0_adv, _, _, _, _, _ = moroz_reintegration_tracking(U0, vx0, vy0)
    U0_new = jnp.clip(U0_adv, 0.0, 1.0)

    # ── Predator ──────────────────────────────────────────────────────────────
    U1_conv = jax.vmap(
        lambda k_fft: jnp.fft.irfft2(fft_U1 * k_fft, s=(H, W))
    )(k_ffts_1)

    G1 = jnp.mean(
        2.0 * jnp.exp(-((U1_conv - mu1) ** 2) / (2.0 * sigma1 ** 2 + 1e-9)) - 1.0,
        axis=0,
    )
    G1_eff = G1 * (1.0 + lambda1 * jnp.clip(U0, 0.0, 1.0))

    dG1_x, dG1_y = compute_sobel_gradients(G1_eff)
    dU1_x, dU1_y = compute_sobel_gradients(U1)

    vx1 = jnp.tanh(v_scale1 * ((1.0 - alpha1) * dG1_x - alpha1 * dU1_x) + chi1 * dscent0_x)
    vy1 = jnp.tanh(v_scale1 * ((1.0 - alpha1) * dG1_y - alpha1 * dU1_y) + chi1 * dscent0_y)

    U1_adv, _, _, _, _, _ = moroz_reintegration_tracking(U1, vx1, vy1)
    U1_new = jnp.clip(U1_adv - gamma1 * U1, 0.0, 1.0)

    return U0_new, U1_new
