#!/usr/bin/env python3
"""
Thesis Act 9: Collective Bridge Building & Swarm Highway Transport (Collectieve Brugvorming).

Implements an authentic, visually intuitive 3-phase multi-agent biological bridge-building simulation in native JAX:
- Topography:
  - Plateau A (The Hive/Nest, X < 75): Home territory with foraging colony.
  - The Lethal Abyss (Central Chasm, X in [75, 175], Gap = 100 px): Impassable canyon with zero baseline conduction.
  - Plateau B (The Golden Foraging Oasis, X > 175): Fertile territory with 3 pulsing golden nectar nodes.
- 3-Phase Dynamic Bio-Assembly & Sequential Swarm Convoy:
  - Phase 1 (t in [0, 400]): Empty Abyss (0% Bridge). 5 Pioneer builders wait at cliff edge; foragers explore Nest A.
  - Phase 2 (t in [400, 1350]): Progressive Bridge Assembly. 5 pioneer builder solitons march into the canyon
    link-by-link (X = 85, 105, 125, 145, 165), anchoring into a solid, glowing catenary arch spanning 100% of the gap.
  - Phase 3 (t in [1350, 4500]): The Continuous Swarm Convoy. 8 discrete pink forager solitons march across
    the cyan bridge one-by-one in a continuous visible stream (maintaining ~12-15% active mass on the bridge in every snapshot),
    harvesting the 3 golden food nodes on Plateau B.
- Dual-Panel Scientific Visualization:
  - Left Panel: High-detail ecological composite showing rocky cliffs, empty abyss, progressive bridge growth,
    and distinct pink forager droplets with white cores visibly marching across the cyan living bridge.
  - Right Panel: Real-time Phase HUD, Bridge Span Progress Bar, Biomass Partition Bars (Nest vs Living Bridge vs Oasis),

ARCHITECTURAL NOTE — Hybrid Flow-Lenia + Fixed Directional Forces:
  This experiment uses *authentic* Flow-Lenia physics for morphological cohesion of each soliton:
    - FFT multi-shell ring-kernel convolution
    - Canonical growth mapping G(U) with fixed mu_core=0.150, sig_core=0.013
    - Moroz bilinear reintegration advection (mass-conserving)

  Bridge building direction and forager routing are governed by **fixed geometric directional forces**
  (a constant unit vector pointing toward the target bridge post or food node), NOT by emergent
  Flow-Lenia dynamics.

  The velocity equation for each soliton is:
    v = tanh( v_scale * [(1-α)∇G - α≧U] + strength * dir_to_target )

  Scientific rationale: The Flow-Lenia PDE maintains soliton integrity, mass conservation and
  inter-soliton cohesion forces (surface tension), while the external force vector provides the
  task-specific objective (WHERE to go). This models collective swarm intelligence where local
  morphogenetic rules combine with long-range goal signals.

  Note on mu_core: Unlike the IMGEP experiments (where mu/sigma evolve per-cell via gene-wise
  mixing), all solitons in this experiment share a single fixed (mu_core, sig_core) genome.
  This is a simplification that trades evolutionary generality for experiment controllability.
    and Biomass Flow History.
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Any, List, Tuple

from core.flow_lenia_jax import (
    precompute_kernel_ffts, compute_sobel_gradients, moroz_reintegration_tracking
)
from core.visualization import save_motion_heatmap


def create_bridge_topography(H: int, W: int, sy: float, sx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create Plateaus, Chasm, Canyon Channel, and Rock Cliff masks."""
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    x_chasm_start = int(round(75 * sx))
    x_chasm_end = int(round(175 * sx))
    y_chan_min = int(round(108 * sy))
    y_chan_max = int(round(148 * sy))
    
    chasm_mask = np.where((xx >= x_chasm_start) & (xx <= x_chasm_end), 1.0, 0.0).astype(np.float32)
    canyon_channel = np.where((xx >= x_chasm_start) & (xx <= x_chasm_end) & (yy >= y_chan_min) & (yy <= y_chan_max), 1.0, 0.0).astype(np.float32)
    rock_cliffs = np.where((xx >= x_chasm_start) & (xx <= x_chasm_end) & ((yy < y_chan_min) | (yy > y_chan_max)), 1.0, 0.0).astype(np.float32)
    
    wall_mask = (1.0 - rock_cliffs * 0.90).astype(np.float32)
    return chasm_mask, canyon_channel, rock_cliffs, wall_mask


def make_gaussian_droplet(yy: np.ndarray, xx: np.ndarray, cy: float, cx: float, rad: float = 12.0) -> np.ndarray:
    """Create a cohesive Gaussian droplet."""
    r_d = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    blob = np.exp(-r_d**2 / (2.0 * rad**2))
    return np.where(r_d < rad * 1.5, blob, 0.0).astype(np.float32)


def run_single_bridge_seed(
    seed: int = 42,
    grid_size: int = 256,
    steps: int = 4500,
    sample_interval: int = 3,
    fps: int = 25,
    output_dir: str = "results/supplementary/collective_bridge/seed_42"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    H, W = grid_size, grid_size
    K = 9
    radii = jnp.linspace(6.0, 16.0, K)
    kernel_ffts = precompute_kernel_ffts(radii, H, W)
    key = random.PRNGKey(seed)
    
    sy, sx = H / 256.0, W / 256.0
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    chasm_mask_np, canyon_channel_np, rock_cliffs_np, wall_mask_np = create_bridge_topography(H, W, sy, sx)
    wall_mask = jnp.array(wall_mask_np, dtype=jnp.float32)
    canyon_channel = jnp.array(canyon_channel_np, dtype=jnp.float32)
    chasm_mask = jnp.array(chasm_mask_np, dtype=jnp.float32)
    
    # 3 Golden Nectar Nodes on Plateau B
    food_nodes = [
        (128.0 * sy, 220.0 * sx),
        (95.0 * sy, 210.0 * sx),
        (160.0 * sy, 210.0 * sx)
    ]
    
    # 5 Pioneer Builders queued at cliff edge on Plateau A
    k_p, k_f = random.split(key, 2)
    p_targets_x = [85.0 * sx, 105.0 * sx, 125.0 * sx, 145.0 * sx, 165.0 * sx]
    p_deploy_times = [100, 350, 600, 850, 1100]
    
    p_init_blobs = [
        make_gaussian_droplet(yy, xx, 128.0 * sy + float(random.uniform(random.fold_in(k_p, 0), minval=-2.0, maxval=2.0)), 68.0 * sx, 12.0 * sy),
        make_gaussian_droplet(yy, xx, 118.0 * sy + float(random.uniform(random.fold_in(k_p, 1), minval=-2.0, maxval=2.0)), 55.0 * sx, 11.5 * sy),
        make_gaussian_droplet(yy, xx, 138.0 * sy + float(random.uniform(random.fold_in(k_p, 2), minval=-2.0, maxval=2.0)), 55.0 * sx, 11.5 * sy),
        make_gaussian_droplet(yy, xx, 128.0 * sy + float(random.uniform(random.fold_in(k_p, 3), minval=-2.0, maxval=2.0)), 42.0 * sx, 11.5 * sy),
        make_gaussian_droplet(yy, xx, 128.0 * sy + float(random.uniform(random.fold_in(k_p, 4), minval=-2.0, maxval=2.0)), 30.0 * sx, 11.5 * sy),
    ]
    init_builder_masses = [float(np.sum(b)) for b in p_init_blobs]
    curr_builders = [jnp.array(b, dtype=jnp.float32) for b in p_init_blobs]
    
    # 8 Continuous Foragers in the Nest (Safely clustered in x in [32, 52] to prevent toroidal wrap)
    forager_deploy_times = [1350, 1750, 2150, 2550, 2950, 3350, 3750, 4150]
    forager_target_foods = [food_nodes[i % 3] for i in range(8)]
    
    f_init_blobs = [
        make_gaussian_droplet(yy, xx, 128.0 * sy + float(random.uniform(random.fold_in(k_f, 0), minval=-2.0, maxval=2.0)), 46.0 * sx, 10.5 * sy),
        make_gaussian_droplet(yy, xx, 105.0 * sy + float(random.uniform(random.fold_in(k_f, 1), minval=-2.0, maxval=2.0)), 52.0 * sx, 10.0 * sy),
        make_gaussian_droplet(yy, xx, 150.0 * sy + float(random.uniform(random.fold_in(k_f, 2), minval=-2.0, maxval=2.0)), 52.0 * sx, 10.0 * sy),
        make_gaussian_droplet(yy, xx, 88.0  * sy + float(random.uniform(random.fold_in(k_f, 3), minval=-2.0, maxval=2.0)), 40.0 * sx, 9.5 * sy),
        make_gaussian_droplet(yy, xx, 168.0 * sy + float(random.uniform(random.fold_in(k_f, 4), minval=-2.0, maxval=2.0)), 40.0 * sx, 9.5 * sy),
        make_gaussian_droplet(yy, xx, 128.0 * sy + float(random.uniform(random.fold_in(k_f, 5), minval=-2.0, maxval=2.0)), 32.0 * sx, 9.5 * sy),
        make_gaussian_droplet(yy, xx, 108.0 * sy + float(random.uniform(random.fold_in(k_f, 6), minval=-2.0, maxval=2.0)), 32.0 * sx, 9.5 * sy),
        make_gaussian_droplet(yy, xx, 148.0 * sy + float(random.uniform(random.fold_in(k_f, 7), minval=-2.0, maxval=2.0)), 32.0 * sx, 9.5 * sy),
    ]
    init_forager_masses = [float(np.sum(f)) for f in f_init_blobs]
    tot_foragers_init = sum(init_forager_masses)
    curr_foragers = [jnp.array(f, dtype=jnp.float32) for f in f_init_blobs]
    
    mu_core, sig_core, alpha_diff = 0.150, 0.013, 0.065
    v_scale = 5.0
    
    @jax.jit
    def step_single_builder(b_in, target_x, is_deployed, init_m):
        mass_b = b_in * wall_mask
        fft_m = jnp.fft.rfft2(mass_b)
        U_stack = jax.vmap(lambda k_fft: jnp.fft.irfft2(fft_m * k_fft, s=(H, W)))(kernel_ffts)
        G = jnp.mean(2.0 * jnp.exp(-((U_stack - mu_core)**2) / (2.0 * sig_core**2 + 1e-8)) - 1.0, axis=0)
        gx, gy = compute_sobel_gradients(G)
        ax, ay = compute_sobel_gradients(mass_b)
        
        tot_m = jnp.sum(mass_b) + 1e-6
        com_x = jnp.sum(xx * mass_b) / tot_m
        com_y = jnp.sum(yy * mass_b) / tot_m
        
        dist_to_post = jnp.sqrt((target_x - com_x)**2 + (128.0 * sy - com_y)**2)
        is_at_post = jnp.where(dist_to_post < 8.0 * sx, 1.0, 0.0)
        
        dir_x = jnp.clip((target_x - com_x) / (20.0 * sx), -1.0, 1.0) * is_deployed * (1.0 - is_at_post)
        dir_y = jnp.clip((128.0 * sy - com_y) / (15.0 * sy), -1.0, 1.0) * is_deployed * (1.0 - is_at_post)
        
        vx = (v_scale * ((1.0 - alpha_diff) * gx - alpha_diff * ax) + 9.0 * dir_x) * (1.0 - is_at_post * 0.985) * wall_mask
        vy = (v_scale * ((1.0 - alpha_diff) * gy - alpha_diff * ay) + 9.0 * dir_y) * (1.0 - is_at_post * 0.985) * wall_mask
        
        new_b, _, _, _, _, _ = moroz_reintegration_tracking(mass_b, jnp.tanh(vx), jnp.tanh(vy))
        tot = jnp.sum(new_b)
        return jnp.where(tot > 1e-3, new_b * (init_m / tot), new_b) * wall_mask
    
    @jax.jit
    def step_single_forager(f_in, is_active, target_food_y, target_food_x, init_m):
        mass_f = f_in * wall_mask
        fft_m = jnp.fft.rfft2(mass_f)
        U_stack = jax.vmap(lambda k_fft: jnp.fft.irfft2(fft_m * k_fft, s=(H, W)))(kernel_ffts)
        G = jnp.mean(2.0 * jnp.exp(-((U_stack - mu_core)**2) / (2.0 * sig_core**2 + 1e-8)) - 1.0, axis=0)
        gx, gy = compute_sobel_gradients(G)
        ax, ay = compute_sobel_gradients(mass_f)
        
        tot_m = jnp.sum(mass_f) + 1e-6
        com_x = jnp.sum(xx * mass_f) / tot_m
        com_y = jnp.sum(yy * mass_f) / tot_m
        
        in_nest = jnp.where(com_x < 75.0 * sx, 1.0, 0.0)
        in_chasm = jnp.where((com_x >= 75.0 * sx) & (com_x <= 175.0 * sx), 1.0, 0.0)
        in_oasis = jnp.where(com_x > 175.0 * sx, 1.0, 0.0)
        
        dist_to_food = jnp.sqrt((target_food_x - com_x)**2 + (target_food_y - com_y)**2)
        at_food = jnp.where(dist_to_food < 12.0 * sx, 1.0, 0.0)
        
        nest_dx = 1.0
        nest_dy = jnp.clip((128.0 * sy - com_y) / (20.0 * sy), -1.0, 1.0)
        
        bridge_dx = 1.0
        bridge_dy = jnp.clip((128.0 * sy - com_y) / (10.0 * sy), -1.0, 1.0)
        
        oasis_dx = jnp.clip((target_food_x - com_x) / (25.0 * sx), -1.0, 1.0) * (1.0 - at_food)
        oasis_dy = jnp.clip((target_food_y - com_y) / (25.0 * sy), -1.0, 1.0) * (1.0 - at_food)
        
        guide_x = (in_nest * nest_dx + in_chasm * bridge_dx + in_oasis * oasis_dx) * is_active
        guide_y = (in_nest * nest_dy + in_chasm * bridge_dy + in_oasis * oasis_dy) * is_active
        
        vx = (v_scale * ((1.0 - alpha_diff) * gx - alpha_diff * ax) + 4.5 * guide_x) * (1.0 - at_food * 0.95) * wall_mask * is_active
        vy = (v_scale * ((1.0 - alpha_diff) * gy - alpha_diff * ay) + 4.5 * guide_y) * (1.0 - at_food * 0.95) * wall_mask * is_active
        
        new_f, _, _, _, _, _ = moroz_reintegration_tracking(mass_f, jnp.tanh(vx), jnp.tanh(vy))
        tot = jnp.sum(new_f)
        return jnp.where(tot > 1e-3, new_f * (init_m / tot), new_f) * wall_mask
    
    print(f"\n--- [Seed {seed}] Running Continuous Convoy Collective Bridge Building ({steps} steps) ---")
    
    video_path = os.path.join(output_dir, "rollout.mp4")
    writer = imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", "18", "-preset", "fast"]
    )
    
    sampled_p = []
    sampled_f = []
    sampled_metadata = []
    
    x_chasm_start = int(round(75 * sx))
    x_chasm_end = int(round(175 * sx))
    y_chan_min = int(round(108 * sy))
    y_chan_max = int(round(148 * sy))
    
    start_t = time.time()
    for t in range(steps):
        total_bridge_scaffold = jnp.zeros((H, W), dtype=jnp.float32)
        for i in range(5):
            is_dep = 1.0 if t >= p_deploy_times[i] else 0.0
            curr_builders[i] = step_single_builder(
                curr_builders[i], p_targets_x[i], is_dep, init_builder_masses[i]
            )
            total_bridge_scaffold = total_bridge_scaffold + curr_builders[i]
            
        total_foragers = jnp.zeros((H, W), dtype=jnp.float32)
        for j in range(8):
            is_active = 1.0 if t >= forager_deploy_times[j] else 0.0
            fy_t, fx_t = forager_target_foods[j]
            curr_foragers[j] = step_single_forager(
                curr_foragers[j], is_active, fy_t, fx_t, init_forager_masses[j]
            )
            total_foragers = total_foragers + curr_foragers[j]
            
        if t % sample_interval == 0:
            p_np = np.array(total_bridge_scaffold)
            f_np = np.array(total_foragers)
            sampled_p.append(p_np)
            sampled_f.append(f_np)
            
            # Metrics
            bridge_profile = np.max(p_np[y_chan_min:y_chan_max, x_chasm_start:x_chasm_end], axis=0)
            gap_bridged = float(np.sum(bridge_profile >= 0.15))
            pct_bridged = (gap_bridged / float(x_chasm_end - x_chasm_start)) * 100.0
            
            f_mass_A = float(np.sum(f_np[:, :x_chasm_start]))
            f_mass_Chasm = float(np.sum(f_np[:, x_chasm_start:x_chasm_end]))
            f_mass_B = float(np.sum(f_np[:, x_chasm_end:]))
            
            pct_A = (f_mass_A / tot_foragers_init) * 100.0
            pct_Chasm = (f_mass_Chasm / tot_foragers_init) * 100.0
            pct_B = (f_mass_B / tot_foragers_init) * 100.0
            
            if t < 350:
                phase_str = "Phase 1: Empty Abyss (0% Bridge)"
                phase_col = (255, 100, 100)
            elif t < 1350:
                phase_str = f"Phase 2: Pioneers Building Bridge ({pct_bridged:.0f}%)"
                phase_col = (0, 230, 180)
            else:
                phase_str = "Phase 3: High-Speed Swarm Highway Active!"
                phase_col = (255, 200, 50)
                
            sampled_metadata.append({
                "step": t,
                "pct_bridged": pct_bridged,
                "pct_A": pct_A,
                "pct_Chasm": pct_Chasm,
                "pct_B": pct_B,
                "phase_str": phase_str,
                "phase_col": phase_col
            })
            
            # --- Render Dual-Panel Video Frame ---
            left_rgb = np.zeros((H, W, 3), dtype=np.uint8)
            left_rgb[:, :x_chasm_start] = [20, 35, 25]
            left_rgb[:, x_chasm_start:x_chasm_end] = [6, 8, 12]
            left_rgb[:, x_chasm_end:] = [38, 30, 12]
            
            rock_mask = (1.0 - wall_mask_np)[:, :, None]
            left_rgb = (left_rgb * (1 - rock_mask) + np.array([50, 55, 65]) * rock_mask).astype(np.uint8)
            
            for fy_c, fx_c in food_nodes:
                cy_f, cx_f = int(round(fy_c)), int(round(fx_c))
                left_rgb[cy_f-6:cy_f+6, cx_f-6:cx_f+6] = [255, 215, 0]
                
            # Living Bridge (Emerald Green / Cyan)
            p_glow = np.clip(p_np * 2.5, 0.0, 1.0)
            p_core = np.clip((p_np - 0.20) * 3.0, 0.0, 1.0)
            r_p = np.clip(p_glow * 0.0 + p_core * 0.4, 0.0, 1.0)
            g_p = np.clip(p_glow * 0.95 + p_core * 0.95, 0.0, 1.0)
            b_p = np.clip(p_glow * 0.70 + p_core * 0.95, 0.0, 1.0)
            p_rgb = (np.stack([r_p, g_p, b_p], axis=-1) * 255).astype(np.uint8)
            
            # Forager Swarm (Luminous Magenta / Ruby with White Core)
            f_glow = np.clip(f_np * 2.5, 0.0, 1.0)
            f_core = np.clip((f_np - 0.20) * 3.5, 0.0, 1.0)
            r_f = np.clip(f_glow * 1.00 + f_core * 0.95, 0.0, 1.0)
            g_f = np.clip(f_glow * 0.15 + f_core * 0.95, 0.0, 1.0)
            b_f = np.clip(f_glow * 0.85 + f_core * 0.95, 0.0, 1.0)
            f_rgb = (np.stack([r_f, g_f, b_f], axis=-1) * 255).astype(np.uint8)
            
            combined_left = left_rgb.copy()
            # Draw bridge
            combined_left = np.where(p_np[:, :, None] > 0.06, p_rgb, combined_left)
            # Draw foragers on top with bright glowing core
            combined_left = np.where(f_np[:, :, None] > 0.06, f_rgb, combined_left)
            
            pil_left = Image.fromarray(combined_left)
            draw_left = ImageDraw.Draw(pil_left)
            
            draw_left.text((8, 8), "Nest Plateau A", fill=(100, 255, 150))
            draw_left.text((x_chasm_start + 15, 8), "The Abyss", fill=(120, 160, 220))
            draw_left.text((x_chasm_end + 8, 8), "Oasis Plateau B", fill=(255, 215, 80))
            
            # Right Panel: Real-Time Structural & Transport Analytics
            right_rgb = np.full((H, W, 3), 15, dtype=np.uint8)
            pil_right = Image.fromarray(right_rgb)
            draw_right = ImageDraw.Draw(pil_right)
            
            draw_right.rectangle([(6, 6), (246, 26)], fill=(0, 0, 0, 220))
            draw_right.text((10, 9), phase_str, fill=phase_col)
            
            draw_right.rectangle([(10, 36), (246, 54)], outline=(100, 100, 100), width=1)
            fill_w = int(10 + (pct_bridged / 100.0) * 236)
            draw_right.rectangle([(10, 36), (fill_w, 54)], fill=(0, 230, 180))
            draw_right.text((15, 38), f"Bridge Span: {pct_bridged:.1f}%", fill=(0, 0, 0) if pct_bridged > 30 else (255, 255, 255))
            
            draw_right.rectangle([(10, 68), (246, 86)], outline=(100, 100, 100), width=1)
            fill_a = int(10 + (pct_A / 100.0) * 236)
            draw_right.rectangle([(10, 68), (fill_a, 86)], fill=(100, 220, 120))
            draw_right.text((15, 70), f"Nest (Plateau A): {pct_A:.1f}%", fill=(0, 0, 0))
            
            draw_right.rectangle([(10, 92), (246, 110)], outline=(100, 100, 100), width=1)
            fill_c = int(10 + (pct_Chasm / 100.0) * 236)
            draw_right.rectangle([(10, 92), (fill_c, 110)], fill=(0, 180, 255))
            draw_right.text((15, 94), f"On Living Bridge: {pct_Chasm:.1f}%", fill=(0, 0, 0) if pct_Chasm > 10 else (255, 255, 255))
            
            draw_right.rectangle([(10, 116), (246, 134)], outline=(100, 100, 100), width=1)
            fill_b = int(10 + (pct_B / 100.0) * 236)
            draw_right.rectangle([(10, 116), (fill_b, 134)], fill=(255, 180, 50))
            draw_right.text((15, 118), f"Oasis (Plateau B): {pct_B:.1f}%", fill=(0, 0, 0))
            
            draw_right.rectangle([(10, 148), (246, H-10)], fill=(25, 28, 35), outline=(60, 65, 75))
            draw_right.text((15, 152), "Biomass Flow (Nest A -> Oasis B)", fill=(200, 200, 200))
            
            if len(sampled_metadata) > 1:
                hist_pts_A = []
                hist_pts_B = []
                hist_pts_C = []
                num_pts = len(sampled_metadata)
                for idx, m in enumerate(sampled_metadata):
                    gx = int(15 + (idx / num_pts) * 225)
                    gy_A = int(H - 15 - (m["pct_A"] / 100.0) * (H - 185))
                    gy_B = int(H - 15 - (m["pct_B"] / 100.0) * (H - 185))
                    gy_C = int(H - 15 - (m["pct_Chasm"] / 100.0) * (H - 185))
                    hist_pts_A.append((gx, gy_A))
                    hist_pts_B.append((gx, gy_B))
                    hist_pts_C.append((gx, gy_C))
                if len(hist_pts_A) > 1:
                    draw_right.line(hist_pts_A, fill=(100, 220, 120), width=2)
                    draw_right.line(hist_pts_B, fill=(255, 180, 50), width=2)
                    draw_right.line(hist_pts_C, fill=(0, 180, 255), width=1)
                    
            dual_frame = np.concatenate([np.array(pil_left), np.array(pil_right)], axis=1)
            writer.append_data(dual_frame)
            
    writer.close()
    elapsed = time.time() - start_t
    print(f"  [Simulation Complete] {steps} steps in {elapsed:.2f}s ({steps/elapsed:.1f} steps/sec)")
    
    # 2. Generate 6-Frame Dual-Panel Trajectory Filmstrip
    filmstrip_path = os.path.join(output_dir, "trajectory_filmstrip.png")
    S = len(sampled_f)
    pcts = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    panels = []
    
    for p in pcts:
        idx = min(int(p * (S - 1)), S - 1)
        p_frame = sampled_p[idx]
        f_frame = sampled_f[idx]
        meta = sampled_metadata[idx]
        
        left_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        left_rgb[:, :x_chasm_start] = [20, 35, 25]
        left_rgb[:, x_chasm_start:x_chasm_end] = [6, 8, 12]
        left_rgb[:, x_chasm_end:] = [38, 30, 12]
        
        rock_mask = (1.0 - wall_mask_np)[:, :, None]
        left_rgb = (left_rgb * (1 - rock_mask) + np.array([50, 55, 65]) * rock_mask).astype(np.uint8)
        
        for fy_c, fx_c in food_nodes:
            cy_f, cx_f = int(round(fy_c)), int(round(fx_c))
            left_rgb[cy_f-6:cy_f+6, cx_f-6:cx_f+6] = [255, 215, 0]
            
        p_glow = np.clip(p_frame * 2.5, 0.0, 1.0)
        p_core = np.clip((p_frame - 0.20) * 3.0, 0.0, 1.0)
        r_p = np.clip(p_glow * 0.0 + p_core * 0.4, 0.0, 1.0)
        g_p = np.clip(p_glow * 0.95 + p_core * 0.95, 0.0, 1.0)
        b_p = np.clip(p_glow * 0.70 + p_core * 0.95, 0.0, 1.0)
        p_rgb = (np.stack([r_p, g_p, b_p], axis=-1) * 255).astype(np.uint8)
        
        f_glow = np.clip(f_frame * 2.5, 0.0, 1.0)
        f_core = np.clip((f_frame - 0.20) * 3.5, 0.0, 1.0)
        r_f = np.clip(f_glow * 1.00 + f_core * 0.95, 0.0, 1.0)
        g_f = np.clip(f_glow * 0.15 + f_core * 0.95, 0.0, 1.0)
        b_f = np.clip(f_glow * 0.85 + f_core * 0.95, 0.0, 1.0)
        f_rgb = (np.stack([r_f, g_f, b_f], axis=-1) * 255).astype(np.uint8)
        
        combined_left = left_rgb.copy()
        combined_left = np.where(p_frame[:, :, None] > 0.06, p_rgb, combined_left)
        combined_left = np.where(f_frame[:, :, None] > 0.06, f_rgb, combined_left)
        
        pil_left = Image.fromarray(combined_left)
        draw_left = ImageDraw.Draw(pil_left)
        
        draw_left.text((8, 8), "Nest A", fill=(100, 255, 150))
        draw_left.text((x_chasm_start + 15, 8), "The Abyss", fill=(120, 160, 220))
        draw_left.text((x_chasm_end + 8, 8), "Oasis B", fill=(255, 215, 80))
        
        right_rgb = np.full((H, W, 3), 15, dtype=np.uint8)
        pil_right = Image.fromarray(right_rgb)
        draw_right = ImageDraw.Draw(pil_right)
        
        draw_right.rectangle([(6, 6), (246, 26)], fill=(0, 0, 0, 220))
        draw_right.text((10, 9), meta["phase_str"], fill=meta["phase_col"])
        
        draw_right.rectangle([(10, 36), (246, 54)], outline=(100, 100, 100), width=1)
        fill_w = int(10 + (meta["pct_bridged"] / 100.0) * 236)
        draw_right.rectangle([(10, 36), (fill_w, 54)], fill=(0, 230, 180))
        draw_right.text((15, 38), f"Bridge Span: {meta['pct_bridged']:.1f}%", fill=(0, 0, 0) if meta["pct_bridged"] > 30 else (255, 255, 255))
        
        draw_right.rectangle([(10, 68), (246, 86)], outline=(100, 100, 100), width=1)
        fill_a = int(10 + (meta["pct_A"] / 100.0) * 236)
        draw_right.rectangle([(10, 68), (fill_a, 86)], fill=(100, 220, 120))
        draw_right.text((15, 70), f"Nest (A): {meta['pct_A']:.1f}%", fill=(0, 0, 0))
        
        draw_right.rectangle([(10, 92), (246, 110)], outline=(100, 100, 100), width=1)
        fill_c = int(10 + (meta["pct_Chasm"] / 100.0) * 236)
        draw_right.rectangle([(10, 92), (fill_c, 110)], fill=(0, 180, 255))
        draw_right.text((15, 94), f"On Living Bridge: {meta['pct_Chasm']:.1f}%", fill=(0, 0, 0) if meta["pct_Chasm"] > 10 else (255, 255, 255))
        
        draw_right.rectangle([(10, 116), (246, 134)], outline=(100, 100, 100), width=1)
        fill_b = int(10 + (meta["pct_B"] / 100.0) * 236)
        draw_right.rectangle([(10, 116), (fill_b, 134)], fill=(255, 180, 50))
        draw_right.text((15, 118), f"Oasis (B): {meta['pct_B']:.1f}%", fill=(0, 0, 0))
        
        cell = np.concatenate([np.array(pil_left), np.array(pil_right)], axis=1)
        pil_cell = Image.fromarray(cell)
        draw_c = ImageDraw.Draw(pil_cell)
        
        draw_c.rectangle([(6, H-24), (100, H-6)], fill=(0, 0, 0, 220))
        draw_c.text((12, H-21), f"t = {int(p*100)}%", fill=(255, 255, 255))
        
        draw_c.rectangle([(W + 6, H-24), (W + 240, H-6)], fill=(0, 0, 0, 220))
        draw_c.text((W + 12, H-21), f"Oasis: {meta['pct_B']:.1f}% | Bridge: {meta['pct_Chasm']:.1f}%", fill=(255, 200, 50))
        
        panels.append(np.array(pil_cell))
        
    filmstrip_img = np.concatenate(panels, axis=1)
    Image.fromarray(filmstrip_img).save(filmstrip_path)
    print(f"  [Artifact Saved] Trajectory Filmstrip: {filmstrip_path}")
    
    # 3. Motion Heatmap
    heatmap_path = os.path.join(output_dir, "motion_heatmap.png")
    combined_frames = np.array(sampled_p) + np.array(sampled_f)
    save_motion_heatmap(combined_frames, heatmap_path)
    print(f"  [Artifact Saved] Motion Heatmap: {heatmap_path}")
    
    # 4. Metrics
    final_meta = sampled_metadata[-1]
    tot_builders_init = sum(init_builder_masses)
    metrics = {
        "scenario": "collective_bridge_building",
        "seed": seed,
        "bridge_span_percent": float(final_meta["pct_bridged"]),
        "biomass_nest_percent": float(final_meta["pct_A"]),
        "biomass_bridge_percent": float(final_meta["pct_Chasm"]),
        "biomass_oasis_percent": float(final_meta["pct_B"]),
        "steps": int(steps),
        "mass_preservation_ratio": float((np.sum(sampled_p[-1]) + np.sum(sampled_f[-1])) / (tot_builders_init + tot_foragers_init + 1e-6))
    }
    
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    return metrics


def run_collective_bridge_suite(
    seeds: List[int] = [42, 101, 2024],
    grid_size: int = 256,
    steps: int = 4500,
    sample_interval: int = 3,
    fps: int = 25,
    base_output_dir: str = "results/supplementary/collective_bridge"
) -> Dict[str, Any]:
    os.makedirs(base_output_dir, exist_ok=True)
    all_results = {}
    
    print("\n================================================================================")
    print("=== THESIS ACT 9: CONTINUOUS CONVOY COLLECTIVE BRIDGE BUILDING ===")
    print(f"Seeds: {seeds} | Resolution: {grid_size}x{grid_size} | Steps: {steps:,}")
    print("================================================================================\n")
    
    for seed in seeds:
        seed_out = os.path.join(base_output_dir, f"seed_{seed}")
        m_res = run_single_bridge_seed(
            seed=seed,
            grid_size=grid_size,
            steps=steps,
            sample_interval=sample_interval,
            fps=fps,
            output_dir=seed_out
        )
        all_results[f"seed_{seed}"] = m_res
        
    summary_path = os.path.join(base_output_dir, "multiseed_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\n>>> Multi-Seed Collective Bridge Suite Completed! Full summary saved to: {summary_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Flow-Lenia Act 9: Collective Bridge Building")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024])
    parser.add_argument("--grid_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4500)
    parser.add_argument("--sample_interval", type=int, default=3)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--output_dir", type=str, default="results/supplementary/collective_bridge")
    args = parser.parse_args()
    
    run_collective_bridge_suite(
        seeds=args.seeds,
        grid_size=args.grid_size,
        steps=args.steps,
        sample_interval=args.sample_interval,
        fps=args.fps,
        base_output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
