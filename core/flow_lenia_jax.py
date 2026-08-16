import os
import sys

# Ensure CUDA NVCC from venv is accessible for JAX PTX compilation on Blackwell (RTX 5090)
_venv_nvcc = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia", "cuda_nvcc", "bin")
if os.path.isdir(_venv_nvcc) and _venv_nvcc not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_nvcc + ":" + os.environ.get("PATH", "")

import jax
import jax.numpy as jnp
from jax import random
from typing import NamedTuple, Tuple, Dict, Any, Optional

class FlowLeniaParams(NamedTuple):
    """
    Parameter container for FlowLenia physics.
    """
    mu: jnp.ndarray
    sigma: jnp.ndarray
    weights: jnp.ndarray
    dt: float = 0.05
    beta: float = 2.0
    depletion_rate: float = 0.04
    regen_rate: float = 0.01
    v_scale: float = 5.4
    alpha_diffusion: float = 0.055

class FlowLeniaState(NamedTuple):
    """
    State container for FlowLenia.
    - mass: (C, H, W) - Mass density channels
    - mu_map: (K, H, W) - Local spatial mu parameter map
    - sigma_map: (K, H, W) - Local spatial sigma parameter map
    - weights_map: (K, H, W) - Local spatial weight parameter map
    - resource_map: (H, W) - Dynamic environmental resource map M(x,y)
    - genome_id_map: (H, W) - Primary genome ID per cell for activity tracking
    """
    mass: jnp.ndarray
    mu_map: jnp.ndarray
    sigma_map: jnp.ndarray
    weights_map: jnp.ndarray
    resource_map: jnp.ndarray
    genome_id_map: jnp.ndarray

def create_multi_shell_ring_kernel_2d(
    radius: float, H: int, W: int,
    b_shells: Tuple[float, ...] = (1.0, 0.5, 0.33),
    r_peaks: Tuple[float, ...] = (0.5, 0.25, 0.75),
    r_width: float = 0.12
) -> jnp.ndarray:
    """
    Create a multi-shell Gaussian ring kernel (Michel et al. 2025/2026, Plantec et al. 2025).
    Multi-shell kernels generate self-organizing gliders and multi-limbed solitons.
    """
    y = jnp.fft.fftfreq(H) * H
    x = jnp.fft.fftfreq(W) * W
    yy, xx = jnp.meshgrid(y, x, indexing='ij')
    r = jnp.sqrt(yy**2 + xx**2) / (radius + 1e-8)
    
    kernel = jnp.zeros((H, W), dtype=jnp.float32)
    for b_val, r_peak in zip(b_shells, r_peaks):
        shell = b_val * jnp.exp(-((r - r_peak)**2) / (2.0 * r_width**2))
        kernel = kernel + shell
        
    kernel = jnp.where(r <= 1.0, kernel, 0.0)
    k_sum = jnp.sum(kernel)
    kernel = jnp.where(k_sum > 1e-8, kernel / k_sum, kernel)
    return kernel

def precompute_kernel_ffts(radii: jnp.ndarray, H: int, W: int) -> jnp.ndarray:
    """
    Precompute rfft2 representations of K multi-shell ring kernels.
    """
    def _single_k(r):
        k_spatial = create_multi_shell_ring_kernel_2d(r, H, W)
        return jnp.fft.rfft2(k_spatial)
    
    return jax.vmap(_single_k)(radii)

def compute_sobel_gradients(field: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute 2D spatial gradients (grad_x = dF/dx, grad_y = dF/dy) using Sobel filtering with periodic boundary conditions.
    
    Standard spatial grid convention:
    - x increases to the right (column index j -> j+1)
    - y increases downwards (row index i -> i+1)
    
    jnp.roll(field, shift=-1, axis=1) pulls element from (j+1) into position j: field(x+1, y)
    jnp.roll(field, shift=1, axis=1) pulls element from (j-1) into position j: field(x-1, y)
    jnp.roll(field, shift=-1, axis=0) pulls element from (i+1) into position i: field(x, y+1)
    jnp.roll(field, shift=1, axis=0) pulls element from (i-1) into position i: field(x, y-1)
    """
    f_right = jnp.roll(field, shift=-1, axis=1) # field(x+1, y)
    f_left = jnp.roll(field, shift=1, axis=1)   # field(x-1, y)
    f_down = jnp.roll(field, shift=-1, axis=0)  # field(x, y+1)
    f_up = jnp.roll(field, shift=1, axis=0)    # field(x, y-1)
    
    f_up_left = jnp.roll(f_up, shift=1, axis=1)
    f_up_right = jnp.roll(f_up, shift=-1, axis=1)
    f_down_left = jnp.roll(f_down, shift=1, axis=1)
    f_down_right = jnp.roll(f_down, shift=-1, axis=1)
    
    # dF/dx = (f(x+1) - f(x-1)) / 2 with Sobel smoothing in y
    grad_x = ((f_up_right + 2.0 * f_right + f_down_right) - (f_up_left + 2.0 * f_left + f_down_left)) / 8.0
    # dF/dy = (f(y+1) - f(y-1)) / 2 with Sobel smoothing in x
    grad_y = ((f_down_left + 2.0 * f_down + f_down_right) - (f_up_left + 2.0 * f_up + f_up_right)) / 8.0
    
    return grad_x, grad_y

def moroz_reintegration_tracking(
    mass: jnp.ndarray,
    vx: jnp.ndarray,
    vy: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Moroz (2020) / Plantec (2025) Bilinear Reintegration Tracking:
    Continuous 2D semi-Lagrangian transport with 9-neighbor bilinear splatting.
    Preserves exact mass conservation without artificial axis-aligned damping drag.
    """
    vx = jnp.clip(vx, -1.0, 1.0)
    vy = jnp.clip(vy, -1.0, 1.0)
    
    fx_pos = jnp.maximum(0.0, vx)
    fx_neg = jnp.maximum(0.0, -vx)
    fx_zero = 1.0 - fx_pos - fx_neg
    
    fy_pos = jnp.maximum(0.0, vy)
    fy_neg = jnp.maximum(0.0, -vy)
    fy_zero = 1.0 - fy_pos - fy_neg
    
    # 9-neighbor outgoing mass parcels from cell (y, x)
    m_00 = mass * fy_zero * fx_zero
    m_0R = mass * fy_zero * fx_pos
    m_0L = mass * fy_zero * fx_neg
    m_D0 = mass * fy_pos * fx_zero
    m_U0 = mass * fy_neg * fx_zero
    m_DR = mass * fy_pos * fx_pos
    m_DL = mass * fy_pos * fx_neg
    m_UR = mass * fy_neg * fx_pos
    m_UL = mass * fy_neg * fx_neg
    
    # Incoming mass from 9 neighbors
    new_mass = (
        m_00
        + jnp.roll(m_0R, shift=1, axis=1)
        + jnp.roll(m_0L, shift=-1, axis=1)
        + jnp.roll(m_D0, shift=1, axis=0)
        + jnp.roll(m_U0, shift=-1, axis=0)
        + jnp.roll(m_DR, shift=(1, 1), axis=(0, 1))
        + jnp.roll(m_DL, shift=(1, -1), axis=(0, 1))
        + jnp.roll(m_UR, shift=(-1, 1), axis=(0, 1))
        + jnp.roll(m_UL, shift=(-1, -1), axis=(0, 1))
    )
    
    retained_center = m_00
    f_in_left = jnp.roll(m_0R, shift=1, axis=1)
    f_in_right = jnp.roll(m_0L, shift=-1, axis=1)
    f_in_up = jnp.roll(m_D0, shift=1, axis=0)
    f_in_down = jnp.roll(m_U0, shift=-1, axis=0)
    
    return new_mass, retained_center, f_in_left, f_in_right, f_in_up, f_in_down

def compute_advection_fluxes(mass: jnp.ndarray, vx: jnp.ndarray, vy: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    _, retained_center, f_in_left, f_in_right, f_in_up, f_in_down = moroz_reintegration_tracking(mass, vx, vy)
    return retained_center, f_in_left, f_in_right, f_in_up, f_in_down

def stochastic_gene_wise_mixing(
    rng_key: jnp.ndarray,
    param_map: jnp.ndarray,
    retained_center: jnp.ndarray,
    f_left: jnp.ndarray,
    f_right: jnp.ndarray,
    f_up: jnp.ndarray,
    f_down: jnp.ndarray
) -> jnp.ndarray:
    K, H, W = param_map.shape
    
    fluxes = jnp.stack([retained_center, f_left, f_right, f_up, f_down], axis=0)
    total_in = jnp.sum(fluxes, axis=0, keepdims=True)
    
    p_sources = jnp.where(total_in > 1e-8, fluxes / total_in, jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]).reshape(5, 1, 1))
    
    p_center = param_map
    p_left = jnp.roll(param_map, shift=1, axis=2)
    p_right = jnp.roll(param_map, shift=-1, axis=2)
    p_up = jnp.roll(param_map, shift=1, axis=1)
    p_down = jnp.roll(param_map, shift=-1, axis=1)
    candidates = jnp.stack([p_center, p_left, p_right, p_up, p_down], axis=0)
    
    gumbel_noise = random.gumbel(rng_key, shape=(5, K, H, W))
    log_p = jnp.log(p_sources + 1e-10)[:, None, :, :]
    
    selected_source_idx = jnp.argmax(log_p + gumbel_noise, axis=0)
    
    cand_flat = candidates.reshape(5, -1)
    idx_flat = selected_source_idx.reshape(-1)
    col_flat = jnp.arange(K * H * W)
    
    sampled_flat = cand_flat[idx_flat, col_flat]
    return sampled_flat.reshape(K, H, W)

def negotiation_mixing(
    param_map: jnp.ndarray,
    G_map: jnp.ndarray,
    retained_center: jnp.ndarray,
    f_left: jnp.ndarray,
    f_right: jnp.ndarray,
    f_up: jnp.ndarray,
    f_down: jnp.ndarray,
    beta: float = 3.0
) -> jnp.ndarray:
    """
    Canonical Negotiation Rule (Plantec et al. 2025):
    Territorial competition where incoming parameter parcels compete based on growth affinity G(U).
    """
    K, H, W = param_map.shape
    
    fluxes = jnp.stack([retained_center, f_left, f_right, f_up, f_down], axis=0) # (5, H, W)
    
    G_center = G_map
    G_left = jnp.roll(G_map, shift=1, axis=1)
    G_right = jnp.roll(G_map, shift=-1, axis=1)
    G_up = jnp.roll(G_map, shift=1, axis=0)
    G_down = jnp.roll(G_map, shift=-1, axis=0)
    G_stack = jnp.stack([G_center, G_left, G_right, G_up, G_down], axis=0) # (5, H, W)
    
    affinity = fluxes * jnp.exp(jnp.clip(beta * G_stack, -10.0, 10.0))
    affinity_sum = jnp.sum(affinity, axis=0, keepdims=True) # (1, H, W)
    
    weights = jnp.where(affinity_sum > 1e-8, affinity / affinity_sum, jnp.array([1.0, 0.0, 0.0, 0.0, 0.0]).reshape(5, 1, 1))
    
    p_center = param_map
    p_left = jnp.roll(param_map, shift=1, axis=2)
    p_right = jnp.roll(param_map, shift=-1, axis=2)
    p_up = jnp.roll(param_map, shift=1, axis=1)
    p_down = jnp.roll(param_map, shift=-1, axis=1)
    candidates = jnp.stack([p_center, p_left, p_right, p_up, p_down], axis=0) # (5, K, H, W)
    
    blended = jnp.sum(weights[:, None, :, :] * candidates, axis=0)
    return blended

def flow_lenia_step_single(
    state: FlowLeniaState,
    kernel_ffts: jnp.ndarray,
    params: FlowLeniaParams,
    rng_key: jnp.ndarray,
    target_mass_total: Optional[float] = None,
    wall_mask: Optional[jnp.ndarray] = None,
    mixing_rule: str = 'gene_wise',
    enable_mutation: bool = False,
    enable_depletion: bool = False,
    mutation_patch_radius: int = 10,
    mutation_std: float = 0.01
) -> FlowLeniaState:
    mass = state.mass
    C, H, W = mass.shape
    K = kernel_ffts.shape[0]
    
    mass_primary = mass[0]
    if wall_mask is not None:
        mass_primary = mass_primary * wall_mask
    
    resource_map = state.resource_map
    if enable_depletion:
        occupied = mass_primary > 0.10
        new_resource = jnp.where(
            occupied,
            jnp.maximum(0.1, resource_map - params.depletion_rate),
            jnp.minimum(1.0, resource_map + params.regen_rate)
        )
    else:
        new_resource = resource_map
        
    fft_m = jnp.fft.rfft2(mass_primary)
    
    def _conv_k(k_fft):
        return jnp.fft.irfft2(fft_m * k_fft, s=(H, W))
    
    U_stack = jax.vmap(_conv_k)(kernel_ffts)
    
    effective_mu = state.mu_map
    sigma_map = state.sigma_map
    weights_map = state.weights_map
    
    # Canonical Flow-Lenia Growth Mapping G_k (Plantec et al. 2025)
    G_k = 2.0 * jnp.exp(-((U_stack - effective_mu)**2) / (2.0 * jnp.square(sigma_map) + 1e-8)) - 1.0
    
    if C > 1:
        repulsion_term = 0.5 * mass[1]
        G = jnp.sum(weights_map * G_k, axis=0) - repulsion_term
    else:
        G = jnp.sum(weights_map * G_k, axis=0)
        
    if enable_depletion:
        G = G * (0.5 + 0.5 * new_resource)
        
    gx, gy = compute_sobel_gradients(G)
    ax, ay = compute_sobel_gradients(mass_primary)
    
    alpha = params.alpha_diffusion
    v_s = params.v_scale
    
    vx = v_s * ((1.0 - alpha) * gx - alpha * ax)
    vy = v_s * ((1.0 - alpha) * gy - alpha * ay)
    
    # Smooth continuous velocity bounded in [-1.0, 1.0] without hard clipping shocks
    vx = jnp.tanh(vx)
    vy = jnp.tanh(vy)
    
    if wall_mask is not None:
        vx = vx * wall_mask
        vy = vy * wall_mask
    
    new_mass_primary, retained_center, f_left, f_right, f_up, f_down = moroz_reintegration_tracking(mass_primary, vx, vy)
    
    if wall_mask is not None:
        new_mass_primary = new_mass_primary * wall_mask
    
    new_mass = mass.at[0].set(new_mass_primary)
    
    rng_key, subkey = random.split(rng_key)
    if mixing_rule == 'gene_wise':
        new_mu_map = stochastic_gene_wise_mixing(subkey, state.mu_map, retained_center, f_left, f_right, f_up, f_down)
        rng_key, subkey = random.split(rng_key)
        new_sigma_map = stochastic_gene_wise_mixing(subkey, state.sigma_map, retained_center, f_left, f_right, f_up, f_down)
        rng_key, subkey = random.split(rng_key)
        new_weights_map = stochastic_gene_wise_mixing(subkey, state.weights_map, retained_center, f_left, f_right, f_up, f_down)
        rng_key, subkey = random.split(rng_key)
        gid_2d = state.genome_id_map[None, :, :]
        new_gid_map = stochastic_gene_wise_mixing(subkey, gid_2d, retained_center, f_left, f_right, f_up, f_down)[0]
    elif mixing_rule == 'negotiation':
        new_mu_map = negotiation_mixing(state.mu_map, G, retained_center, f_left, f_right, f_up, f_down, beta=params.beta)
        new_sigma_map = negotiation_mixing(state.sigma_map, G, retained_center, f_left, f_right, f_up, f_down, beta=params.beta)
        new_weights_map = negotiation_mixing(state.weights_map, G, retained_center, f_left, f_right, f_up, f_down, beta=params.beta)
        gid_2d = state.genome_id_map[None, :, :]
        p_c = gid_2d[0]
        p_l = jnp.roll(p_c, shift=1, axis=1)
        p_r = jnp.roll(p_c, shift=-1, axis=1)
        p_u = jnp.roll(p_c, shift=1, axis=0)
        p_d = jnp.roll(p_c, shift=-1, axis=0)
        gid_candidates = jnp.stack([p_c, p_l, p_r, p_u, p_d], axis=0)
        
        fluxes = jnp.stack([retained_center, f_left, f_right, f_up, f_down], axis=0)
        G_stack = jnp.stack([G, jnp.roll(G, 1, axis=1), jnp.roll(G, -1, axis=1), jnp.roll(G, 1, axis=0), jnp.roll(G, -1, axis=0)], axis=0)
        affinity = fluxes * jnp.exp(jnp.clip(params.beta * G_stack, -10.0, 10.0))
        top_src = jnp.argmax(affinity, axis=0)
        
        cand_flat = gid_candidates.reshape(5, -1)
        idx_flat = top_src.reshape(-1)
        col_flat = jnp.arange(H * W)
        new_gid_map = cand_flat[idx_flat, col_flat].reshape(H, W)
    else:
        new_mu_map = state.mu_map
        new_sigma_map = state.sigma_map
        new_weights_map = state.weights_map
        new_gid_map = state.genome_id_map
    
    new_mu_map = jnp.clip(new_mu_map, 0.10, 0.28)
    new_sigma_map = jnp.clip(new_sigma_map, 0.008, 0.025)
    new_weights_map = jnp.clip(new_weights_map, 0.0, 2.0)
    
    # Periodic Localized Mutation
    rng_key, subk1, subk2, subk3 = random.split(rng_key, 4)
    cy = random.randint(subk1, (), 0, H)
    cx = random.randint(subk2, (), 0, W)
    
    yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
    dist_sq = (yy - cy)**2 + (xx - cx)**2
    patch_mask = (dist_sq <= mutation_patch_radius**2)[None, :, :]
    
    mu_noise = random.normal(subk3, shape=new_mu_map.shape) * mutation_std
    mutated_mu_map = jnp.where(patch_mask, jnp.clip(new_mu_map + mu_noise, 0.10, 0.28), new_mu_map)
    
    new_mu_map = jnp.where(enable_mutation, mutated_mu_map, new_mu_map)
    
    return FlowLeniaState(
        mass=new_mass,
        mu_map=new_mu_map,
        sigma_map=new_sigma_map,
        weights_map=new_weights_map,
        resource_map=new_resource,
        genome_id_map=new_gid_map
    )

def initialize_multi_patch_state(
    rng_key: jnp.ndarray,
    H: int,
    W: int,
    C: int,
    K: int,
    n_patches: int,
    kernel_radii: jnp.ndarray,
    mu_presets: Optional[jnp.ndarray] = None,
    sigma_presets: Optional[jnp.ndarray] = None,
    weights_presets: Optional[jnp.ndarray] = None,
    wall_mask: Optional[jnp.ndarray] = None,
    is_corridor_test: bool = False
) -> FlowLeniaState:
    mass = jnp.zeros((C, H, W), dtype=jnp.float32)
    mu_map = jnp.full((K, H, W), 0.15, dtype=jnp.float32)
    sigma_map = jnp.full((K, H, W), 0.015, dtype=jnp.float32)
    weights_map = jnp.full((K, H, W), 1.0 / K, dtype=jnp.float32)
    resource_map = jnp.ones((H, W), dtype=jnp.float32)
    genome_id_map = jnp.full((H, W), -1, dtype=jnp.int32)
    
    yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
    center_y, center_x = H // 2, W // 2
    
    for i in range(n_patches):
        rng_key, subk_pos, subk_blobs, subk_angle, subk_p1, subk_p2 = random.split(rng_key, 6)
        
        if is_corridor_test:
            cy = int(H * 0.50 + random.uniform(subk_pos, (), minval=-10.0, maxval=10.0))
            cx = int(W * 0.22 + random.uniform(subk_pos, (), minval=-10.0, maxval=10.0))
            dy_dir = random.uniform(subk_angle, (), minval=-0.20, maxval=0.20)
            dx_dir = random.uniform(subk_angle, (), minval=0.85, maxval=1.00)
        else:
            base_angle = (2.0 * jnp.pi * i) / float(n_patches)
            angle_jitter = random.uniform(subk_pos, (), minval=-0.25, maxval=0.25)
            angle = base_angle + angle_jitter
            min_dim = float(min(H, W))
            if n_patches > 1:
                radius_offset = min_dim * random.uniform(subk_pos, (), minval=0.18, maxval=0.28)
            else:
                radius_offset = 0.0
            cy = int(center_y + radius_offset * jnp.sin(angle))
            cx = int(center_x + radius_offset * jnp.cos(angle))
            
            # Direct swimming vector inward towards the central arena with slight rotational drift
            dir_angle = angle + jnp.pi + random.uniform(subk_angle, (), minval=-0.30, maxval=0.30)
            dy_dir = jnp.sin(dir_angle)
            dx_dir = jnp.cos(dir_angle)
        
        # Multi-Blob Asymmetric Cluster Seeding (Large fleshy cellular organisms with directional momentum)
        n_blobs = int(random.randint(subk_blobs, (), 2, 4))
        patch_density = jnp.zeros((H, W), dtype=jnp.float32)
        patch_mask = jnp.zeros((H, W), dtype=jnp.bool_)
        
        for b in range(n_blobs):
            rng_key, subk_b1, subk_b2, subk_b3 = random.split(rng_key, 4)
            off_y = random.uniform(subk_b1, (), minval=-8.0, maxval=8.0)
            off_x = random.uniform(subk_b2, (), minval=-8.0, maxval=8.0)
            r_blob = random.uniform(subk_b3, (), minval=9.0, maxval=15.0)
            amp = random.uniform(subk_b3, (), minval=0.80, maxval=0.98)
            
            d_sq = (yy - (cy + off_y))**2 + (xx - (cx + off_x))**2
            b_mask = d_sq <= (r_blob * 1.6)**2
            
            rel_y = (yy - cy) / (r_blob + 1e-8)
            rel_x = (xx - cx) / (r_blob + 1e-8)
            density_slope = jnp.clip(1.0 + 1.2 * (rel_y * dy_dir + rel_x * dx_dir), 0.1, 2.5)
            
            b_dens = amp * jnp.exp(-d_sq / (2.0 * (r_blob / 2.0)**2)) * density_slope * b_mask
            patch_density = patch_density + b_dens
            patch_mask = patch_mask | b_mask
            
        patch_density = jnp.clip(patch_density, 0.0, 1.0)
        mass = mass.at[0].add(patch_density)
        
        if mu_presets is not None and i < len(mu_presets):
            patch_mu = mu_presets[i][:, None, None]
        else:
            rng_key, subk_p = random.split(rng_key)
            patch_mu = random.uniform(subk_p, (K, 1, 1), minval=0.13, maxval=0.22)
            
        if sigma_presets is not None and i < len(sigma_presets):
            patch_sigma = sigma_presets[i][:, None, None]
        else:
            rng_key, subk_p = random.split(rng_key)
            patch_sigma = random.uniform(subk_p, (K, 1, 1), minval=0.011, maxval=0.024)
            
        if weights_presets is not None and i < len(weights_presets):
            patch_w = weights_presets[i][:, None, None]
        else:
            patch_w = jnp.full((K, 1, 1), 1.0 / K)
            
        mu_map = jnp.where(patch_mask[None, :, :], patch_mu, mu_map)
        sigma_map = jnp.where(patch_mask[None, :, :], patch_sigma, sigma_map)
        weights_map = jnp.where(patch_mask[None, :, :], patch_w, weights_map)
        genome_id_map = jnp.where(patch_mask, i, genome_id_map)
        
    mass = jnp.clip(mass, 0.0, 1.0)
    
    if wall_mask is not None:
        mass = mass * wall_mask
        
    return FlowLeniaState(
        mass=mass,
        mu_map=mu_map,
        sigma_map=sigma_map,
        weights_map=weights_map,
        resource_map=resource_map,
        genome_id_map=genome_id_map
    )

def run_flow_lenia_rollout(
    state: FlowLeniaState,
    kernel_ffts: jnp.ndarray,
    params: FlowLeniaParams,
    rng_key: jnp.ndarray,
    num_steps: int = 2000,
    sample_interval: int = 250,
    wall_mask: Optional[jnp.ndarray] = None,
    mixing_rule: str = 'gene_wise',
    enable_mutation: bool = True,
    enable_depletion: bool = False,
    mutation_interval: int = 100
) -> Tuple[FlowLeniaState, jnp.ndarray, jnp.ndarray]:
    target_mass = jnp.sum(state.mass[0])
    def _step_fn(carry, step_idx):
        curr_state, key = carry
        key, subkey = random.split(key)
        
        do_mut = enable_mutation & ((step_idx % mutation_interval) == 0)
        next_state = flow_lenia_step_single(
            curr_state,
            kernel_ffts,
            params,
            subkey,
            target_mass_total=target_mass,
            wall_mask=wall_mask,
            mixing_rule=mixing_rule,
            enable_mutation=do_mut,
            enable_depletion=enable_depletion
        )
        return (next_state, key), (next_state.mass, next_state.genome_id_map)
    
    (final_state, _), (all_mass_frames, all_gid_maps) = jax.lax.scan(
        _step_fn,
        (state, rng_key),
        jnp.arange(num_steps)
    )
    
    sample_indices = jnp.arange(0, num_steps, sample_interval)
    sampled_mass_frames = all_mass_frames[sample_indices]
    sampled_genome_maps = all_gid_maps[sample_indices]
    
    return final_state, sampled_mass_frames, sampled_genome_maps
