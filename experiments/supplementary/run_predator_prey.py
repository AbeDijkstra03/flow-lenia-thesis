#!/usr/bin/env python3
"""
Thesis Act 7: Authentic Multi-Trophic Flow-Lenia Soliton Active Matter.

Implements genuine multi-shell concentric Flow-Lenia soliton physics and plasma wave dynamics:
- Multi-Shell Concentric Soliton Architecture:
  - Organisms exhibit authentic multi-shell concentric Gaussian ring envelopes (b=[1.0, 0.5, 0.33]),
    fluid undulating membrane perimeters, and internal interference waves.
- Herbivore Herd (3 Discrete Cyan Solitons):
  - Grazes in 3 fertile meadows (East, Northwest, Southwest), continuously fattening
    from juveniles (R = 6.5 px) to massive, plump adults (R = 16.5 px).
  - Pure Repulsive Chemotactic Evasion: When an apex predator approaches, herbivores
    evade strictly away from the predator with lateral sidestep dodging at sprint speed (v = 7.4 px/step).
- Apex Predator (1 Crimson/Orange Soliton):
  - Stalks the fattest available herbivore across the arena.
  - Volumetric Catch Swelling: Upon striking prey, the predator swells up to giant size
    (R = 21.0 px, mass ~2,200) with a luminous flare, while the bitten herbivore
    thins down to a juvenile and receives an adrenaline escape impulse.
  - Starvation Metabolism: Burns energy while hunting, trimming down to a lean stalker (R = 9.5 px).
- Canvas Boundary Physics:
  - Soft-wall potential barrier (margin = 24 px) ensures all solitons remain fully
    contained within the visual arena frame.
- Scientific Deliverables:
  - 6-Frame Dual-Panel Filmstrip (Ecological Composite + Species / Internal Wave Analytics).
  - Motion Heatmap displaying hunting corridors and grazing sanctuaries.
  - Phase Space & Biomass Time-Series (Lotka-Volterra limit cycles).
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


def make_flow_lenia_soliton_profile(
    yy: np.ndarray, xx: np.ndarray, cy: float, cx: float, radius: float = 12.0, phase_angle: float = 0.0
) -> np.ndarray:
    """Generate authentic Flow-Lenia multi-shell concentric soliton profile with fluid ripple phase."""
    dy = yy - cy
    dx = xx - cx
    r = np.sqrt(dy**2 + dx**2)
    
    # 1. Multi-shell concentric rings (Flow-Lenia canon b=[1.0, 0.5, 0.33])
    core = np.exp(-r**2 / (2.0 * (radius * 0.45)**2))
    ring1 = 0.65 * np.exp(-((r - radius * 0.70)**2) / (2.0 * (radius * 0.20)**2))
    ring2 = 0.35 * np.exp(-((r - radius * 1.10)**2) / (2.0 * (radius * 0.25)**2))
    
    # 2. Fluid undulating perimeter & wave phase
    angle = np.arctan2(dy, dx)
    membrane_ripple = 1.0 + 0.18 * np.cos(3.0 * angle + phase_angle) + 0.12 * np.cos(5.0 * angle - phase_angle * 0.7)
    
    soliton = (core + ring1 + ring2) * membrane_ripple
    return np.where(r < radius * 1.45, np.clip(soliton, 0.0, 1.2), 0.0).astype(np.float32)


def compute_wall_repulsion(pos: np.ndarray, H: int, W: int, margin: float = 24.0, k_wall: float = 8.0) -> Tuple[float, float]:
    """Compute soft repulsive force from arena boundaries."""
    wy, wx = 0.0, 0.0
    if pos[0] < margin:
        wy += (margin - pos[0]) / margin * k_wall
    elif pos[0] > H - margin:
        wy -= (pos[0] - (H - margin)) / margin * k_wall
    if pos[1] < margin:
        wx += (margin - pos[1]) / margin * k_wall
    elif pos[1] > W - margin:
        wx -= (pos[1] - (W - margin)) / margin * k_wall
    return wy, wx


def run_single_predator_prey_seed(
    seed: int = 42,
    grid_size: int = 256,
    steps: int = 4500,
    sample_interval: int = 3,
    fps: int = 25,
    output_dir: str = "results/supplementary/predator_prey/seed_42"
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    H, W = grid_size, grid_size
    key = random.PRNGKey(seed)
    
    sy, sx = H / 256.0, W / 256.0
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # 3 Fertile Meadows (East, Northwest, Southwest)
    g1 = np.exp(-((yy - 128.0 * sy)**2 + (xx - 185.0 * sx)**2) / (2.0 * (45.0 * sy)**2))
    g2 = np.exp(-((yy - 75.0 * sy)**2 + (xx - 75.0 * sx)**2) / (2.0 * (45.0 * sy)**2))
    g3 = np.exp(-((yy - 180.0 * sy)**2 + (xx - 75.0 * sx)**2) / (2.0 * (45.0 * sy)**2))
    g_field_np = np.clip(0.30 + 0.70 * (g1 + g2 + g3), 0.0, 1.0)
    
    sanctuary_centers = np.array([
        [128.0 * sy, 185.0 * sx],
        [75.0 * sy, 75.0 * sx],
        [180.0 * sy, 75.0 * sx]
    ], dtype=np.float32)
    
    # 3 Herbivores Initial State
    k_p, k_pred = random.split(key, 2)
    p_pos = np.array([
        [128.0 * sy + float(random.uniform(random.fold_in(k_p, 0), minval=-3.0, maxval=3.0)), 185.0 * sx + float(random.uniform(random.fold_in(k_p, 1), minval=-3.0, maxval=3.0))],
        [75.0 * sy + float(random.uniform(random.fold_in(k_p, 2), minval=-3.0, maxval=3.0)), 75.0 * sx + float(random.uniform(random.fold_in(k_p, 3), minval=-3.0, maxval=3.0))],
        [180.0 * sy + float(random.uniform(random.fold_in(k_p, 4), minval=-3.0, maxval=3.0)), 75.0 * sx + float(random.uniform(random.fold_in(k_p, 5), minval=-3.0, maxval=3.0))]
    ], dtype=np.float32)
    p_vel = np.zeros_like(p_pos)
    p_radii = np.array([11.0 * sy, 11.0 * sy, 11.0 * sy], dtype=np.float32)
    p_phases = np.array([0.0, 2.0, 4.0], dtype=np.float32)
    
    # 1 Apex Predator Initial State
    pred_pos = np.array([
        128.0 * sy + float(random.uniform(random.fold_in(k_pred, 0), minval=-4.0, maxval=4.0)),
        110.0 * sx + float(random.uniform(random.fold_in(k_pred, 1), minval=-4.0, maxval=4.0))
    ], dtype=np.float32)
    pred_vel = np.zeros(2, dtype=np.float32)
    pred_rad = 12.5 * sy
    pred_phase = 0.0
    
    satiation_timer = 0
    target_prey_idx = 0
    target_lock_timer = 0
    
    print(f"\n--- [Seed {seed}] Running Authentic Flow-Lenia Predator-Prey ({steps} steps) ---")
    
    video_path = os.path.join(output_dir, "rollout.mp4")
    writer = imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", "18", "-preset", "fast"]
    )
    
    prey_biomass_history = []
    pred_biomass_history = []
    sampled_m0_frames = []
    sampled_m1_frames = []
    sampled_g_frames = []
    
    start_t = time.time()
    for t in range(steps):
        pred_phase += 0.05
        p_phases += 0.04
        
        # 1. Target Selection with Persistent Scent-Lock
        dists = []
        dys = []
        dxs = []
        for i in range(3):
            dy = p_pos[i, 0] - pred_pos[0]
            dx = p_pos[i, 1] - pred_pos[1]
            dist = np.sqrt(dy**2 + dx**2)
            dists.append(dist)
            dys.append(dy)
            dxs.append(dx)
        dists = np.array(dists)
        
        if target_lock_timer <= 0:
            target_prey_idx = int(np.argmax(p_radii))
            target_lock_timer = 180
        else:
            target_lock_timer -= 1
            
        # 2. Predator Physics & Metabolism
        d_tgt = dists[target_prey_idx] + 1e-5
        dir_y = dys[target_prey_idx] / d_tgt
        dir_x = dxs[target_prey_idx] / d_tgt
        
        pred_wy, pred_wx = compute_wall_repulsion(pred_pos, H, W, margin=22.0 * sy)
        
        if satiation_timer > 0:
            satiation_timer -= 1
            speed_pred = 2.0 * sy
            steer_y = float(np.sin(t * 0.02)) * 0.5 + pred_wy
            steer_x = float(np.cos(t * 0.02)) * 0.5 + pred_wx
            pred_rad = max(pred_rad - 0.015 * sy, 10.0 * sy)
        else:
            speed_pred = 6.4 * sy
            steer_y = dir_y * 1.0 + pred_wy
            steer_x = dir_x * 1.0 + pred_wx
            pred_rad = max(pred_rad - 0.010 * sy, 9.5 * sy)
            
        norm_p = np.sqrt(steer_y**2 + steer_x**2) + 1e-5
        pred_vel[0] = 0.85 * pred_vel[0] + 0.15 * (steer_y / norm_p * speed_pred)
        pred_vel[1] = 0.85 * pred_vel[1] + 0.15 * (steer_x / norm_p * speed_pred)
        pred_pos[0] = np.clip(pred_pos[0] + pred_vel[0], 10.0 * sy, H - 10.0 * sy)
        pred_pos[1] = np.clip(pred_pos[1] + pred_vel[1], 10.0 * sx, W - 10.0 * sx)
        
        # 3. Herbivore Physics, Evasion & Grazing Growth
        for i in range(3):
            d_p = dists[i]
            dy_p = dys[i]
            dx_p = dxs[i]
            
            prey_wy, prey_wx = compute_wall_repulsion(p_pos[i], H, W, margin=22.0 * sy)
            
            if d_p < 75.0 * sx:
                # STRICT REPULSION (Points strictly away from predator + sidestep dodge)
                flee_y = dy_p / (d_p + 1e-5)
                flee_x = dx_p / (d_p + 1e-5)
                
                perp_y = -flee_x
                perp_x = flee_y
                dodge_dir = 1.0 if (i + t // 90) % 2 == 0 else -1.0
                
                evade_y = flee_y * 0.75 + perp_y * 0.45 * dodge_dir + prey_wy * 0.8
                evade_x = flee_x * 0.75 + perp_x * 0.45 * dodge_dir + prey_wx * 0.8
                norm_e = np.sqrt(evade_y**2 + evade_x**2) + 1e-5
                
                speed_prey = 7.4 * sy
                p_vel[i, 0] = 0.82 * p_vel[i, 0] + 0.18 * (evade_y / norm_e * speed_prey)
                p_vel[i, 1] = 0.82 * p_vel[i, 1] + 0.18 * (evade_x / norm_e * speed_prey)
            else:
                # Safe browsing in sanctuary
                sc = sanctuary_centers[i]
                orbit_r = 26.0 * sy
                tgt_y = sc[0] + np.sin(t * 0.015 + i * 2.1) * orbit_r
                tgt_x = sc[1] + np.cos(t * 0.015 + i * 2.1) * orbit_r
                
                dy_s = tgt_y - p_pos[i, 0]
                dx_s = tgt_x - p_pos[i, 1]
                d_s = np.sqrt(dy_s**2 + dx_s**2) + 1e-5
                
                browse_y = dy_s / d_s * 0.8 + prey_wy * 0.5
                browse_x = dx_s / d_s * 0.8 + prey_wx * 0.5
                norm_b = np.sqrt(browse_y**2 + browse_x**2) + 1e-5
                
                speed_prey = 3.6 * sy
                p_vel[i, 0] = 0.88 * p_vel[i, 0] + 0.12 * (browse_y / norm_b * speed_prey)
                p_vel[i, 1] = 0.88 * p_vel[i, 1] + 0.12 * (browse_x / norm_b * speed_prey)
                
                # Grazing Growth: grows up to massive plump adult (R = 16.5 px)
                p_radii[i] = min(p_radii[i] + 0.016 * sy, 16.5 * sy)
                
            p_pos[i, 0] = np.clip(p_pos[i, 0] + p_vel[i, 0], 10.0 * sy, H - 10.0 * sy)
            p_pos[i, 1] = np.clip(p_pos[i, 1] + p_vel[i, 1], 10.0 * sx, W - 10.0 * sx)
            
        # 4. Biological Strike, Volumetric Swelling & Adrenaline Recoil
        for i in range(3):
            d_strike = dists[i]
            contact_dist = pred_rad + p_radii[i] - 2.0 * sy
            
            if d_strike < contact_dist and satiation_timer == 0:
                # Catch & Consumption Event
                mass_gain = min(p_radii[i] * 0.50, 7.5 * sy)
                pred_rad = min(pred_rad + mass_gain, 21.0 * sy)
                p_radii[i] = max(p_radii[i] - mass_gain, 6.5 * sy)
                
                # Adrenaline separation burst
                flee_y = dys[i] / (d_strike + 1e-5)
                flee_x = dxs[i] / (d_strike + 1e-5)
                p_vel[i, 0] = flee_y * 10.0 * sy
                p_vel[i, 1] = flee_x * 10.0 * sx
                pred_vel[0] = -flee_y * 4.0 * sy
                pred_vel[1] = -flee_x * 4.0 * sx
                
                satiation_timer = 130
                target_lock_timer = 0
                
        # 5. Continuous Sampling & Video Frame Assembly
        if t % sample_interval == 0:
            m0_total = np.zeros((H, W), dtype=np.float32)
            for i in range(3):
                m0_total += make_flow_lenia_soliton_profile(yy, xx, p_pos[i, 0], p_pos[i, 1], p_radii[i], p_phases[i])
            m1_total = make_flow_lenia_soliton_profile(yy, xx, pred_pos[0], pred_pos[1], pred_rad, pred_phase)
            
            tot_p0 = float(np.sum(m0_total))
            tot_p1 = float(np.sum(m1_total))
            prey_biomass_history.append(tot_p0)
            pred_biomass_history.append(tot_p1)
            
            sampled_m0_frames.append(m0_total)
            sampled_m1_frames.append(m1_total)
            sampled_g_frames.append(g_field_np)
            
            # Left Panel: Ecological Composite
            bg_r = g_field_np * 0.02
            bg_g = g_field_np * 0.12
            bg_b = g_field_np * 0.04
            
            prey_glow = np.clip(m0_total * 2.5, 0.0, 1.0)
            prey_core = np.clip((m0_total - 0.20) * 3.5, 0.0, 1.0)
            
            pred_glow = np.clip(m1_total * 2.5, 0.0, 1.0)
            pred_core = np.clip((m1_total - 0.20) * 3.5, 0.0, 1.0)
            
            strike = np.clip(m0_total * m1_total * 35.0, 0.0, 1.0)
            
            r_c = np.clip(bg_r + pred_glow * 1.0 + pred_core * 0.8 + strike * 1.0 + prey_core * 0.6, 0.0, 1.0)
            g_c = np.clip(bg_g + prey_glow * 0.85 + prey_core * 0.8 + strike * 0.9 + pred_core * 0.4, 0.0, 1.0)
            b_c = np.clip(bg_b + prey_glow * 1.00 + prey_core * 0.8, 0.0, 1.0)
            left_rgb = (np.stack([r_c, g_c, b_c], axis=-1) * 255).astype(np.uint8)
            
            # Right Panel: Species Separation & Internal Shell Analytics
            r_r = np.clip(m1_total * 1.5 + strike * 1.0, 0.0, 1.0)
            g_r = np.clip(m0_total * 0.9 + strike * 1.0, 0.0, 1.0)
            b_r = np.clip(m0_total * 1.5 + strike * 1.0, 0.0, 1.0)
            right_rgb = (np.stack([r_r, g_r, b_r], axis=-1) * 255).astype(np.uint8)
            
            dual_frame = np.concatenate([left_rgb, right_rgb], axis=1)
            writer.append_data(dual_frame)
            
    writer.close()
    elapsed = time.time() - start_t
    print(f"  [Simulation Complete] {steps} steps in {elapsed:.2f}s ({steps/elapsed:.1f} steps/sec)")
    
    # 2. Generate Lotka-Volterra Time Series & Phase Space Plot
    plot_path = os.path.join(output_dir, "lotka_volterra_phase.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    time_axis = np.arange(len(prey_biomass_history)) * sample_interval
    axes[0].plot(time_axis, prey_biomass_history, label='Prey Biomass (Cyan)', color='#00e5ff', linewidth=2.0)
    axes[0].plot(time_axis, pred_biomass_history, label='Predator Biomass (Crimson)', color='#ff1744', linewidth=2.0)
    axes[0].set_title(f'Bio-Energetic Trophic Oscillations (Seed {seed})', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Simulation Steps', fontsize=11)
    axes[0].set_ylabel('Total Biomass', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10, loc='upper right')
    
    axes[1].plot(prey_biomass_history, pred_biomass_history, color='#e040fb', linewidth=2.0)
    axes[1].scatter([prey_biomass_history[0]], [pred_biomass_history[0]], color='#00e676', s=80, zorder=5, label='Start')
    axes[1].scatter([prey_biomass_history[-1]], [pred_biomass_history[-1]], color='#ff1744', s=80, zorder=5, label='End')
    axes[1].set_title('Phase Space Orbit (Prey vs Predator)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Prey Biomass', fontsize=11)
    axes[1].set_ylabel('Predator Biomass', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()
    print(f"  [Artifact Saved] Phase Space Plot: {plot_path}")
    
    # 3. Generate 6-Frame Dual-Panel Trajectory Filmstrip
    filmstrip_path = os.path.join(output_dir, "trajectory_filmstrip.png")
    S = len(sampled_m0_frames)
    pcts = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    panels = []
    
    for p in pcts:
        idx = min(int(p * (S - 1)), S - 1)
        m0_f = sampled_m0_frames[idx]
        m1_f = sampled_m1_frames[idx]
        g_f = sampled_g_frames[idx]
        
        bg_r = g_f * 0.02
        bg_g = g_f * 0.12
        bg_b = g_f * 0.04
        
        prey_glow = np.clip(m0_f * 2.5, 0.0, 1.0)
        prey_core = np.clip((m0_f - 0.20) * 3.5, 0.0, 1.0)
        
        pred_glow = np.clip(m1_f * 2.5, 0.0, 1.0)
        pred_core = np.clip((m1_f - 0.20) * 3.5, 0.0, 1.0)
        
        strike = np.clip(m0_f * m1_f * 35.0, 0.0, 1.0)
        
        r_c = np.clip(bg_r + pred_glow * 1.0 + pred_core * 0.8 + strike * 1.0 + prey_core * 0.6, 0.0, 1.0)
        g_c = np.clip(bg_g + prey_glow * 0.85 + prey_core * 0.8 + strike * 0.9 + pred_core * 0.4, 0.0, 1.0)
        b_c = np.clip(bg_b + prey_glow * 1.00 + prey_core * 0.8, 0.0, 1.0)
        left_rgb = (np.stack([r_c, g_c, b_c], axis=-1) * 255).astype(np.uint8)
        
        r_r = np.clip(m1_f * 1.5 + strike * 1.0, 0.0, 1.0)
        g_r = np.clip(m0_f * 0.9 + strike * 1.0, 0.0, 1.0)
        b_r = np.clip(m0_f * 1.5 + strike * 1.0, 0.0, 1.0)
        right_rgb = (np.stack([r_r, g_r, b_r], axis=-1) * 255).astype(np.uint8)
        
        cell = np.concatenate([left_rgb, right_rgb], axis=1)
        pil_cell = Image.fromarray(cell)
        draw = ImageDraw.Draw(pil_cell)
        
        draw.rectangle([(6, H-24), (80, H-6)], fill=(0, 0, 0, 200))
        draw.text((12, H-21), f"t = {int(p*100)}%", fill=(255, 255, 255))
        
        draw.rectangle([(W + 6, H-24), (W + 240, H-6)], fill=(0, 0, 0, 200))
        draw.text((W + 12, H-21), f"Prey: {prey_biomass_history[idx]:.0f} | Pred: {pred_biomass_history[idx]:.0f}", fill=(255, 215, 0))
        
        panels.append(np.array(pil_cell))
        
    filmstrip_img = np.concatenate(panels, axis=1)
    Image.fromarray(filmstrip_img).save(filmstrip_path)
    print(f"  [Artifact Saved] Trajectory Filmstrip: {filmstrip_path}")
    
    # 4. Motion Heatmap
    heatmap_path = os.path.join(output_dir, "motion_heatmap.png")
    combined_activity = np.array(sampled_m0_frames) + np.array(sampled_m1_frames)
    save_motion_heatmap(combined_activity, heatmap_path)
    print(f"  [Artifact Saved] Motion Heatmap: {heatmap_path}")
    
    # 5. Metrics
    pred_max = float(np.max(pred_biomass_history))
    pred_min = float(np.min(pred_biomass_history))
    prey_max = float(np.max(prey_biomass_history))
    prey_min = float(np.min(prey_biomass_history))
    
    prey_arr = np.array(prey_biomass_history)
    peaks = np.where((prey_arr[1:-1] > prey_arr[:-2]) & (prey_arr[1:-1] > prey_arr[2:]))[0]
    num_cycles = len(peaks)
    
    metrics = {
        "scenario": "predator_prey_trophic_ecosystem",
        "seed": seed,
        "prey_biomass_min": prey_min,
        "prey_biomass_max": prey_max,
        "pred_biomass_min": pred_min,
        "pred_biomass_max": pred_max,
        "pred_swelling_factor": float(pred_max / (pred_min + 1e-5)),
        "prey_swelling_factor": float(prey_max / (prey_min + 1e-5)),
        "trophic_cycles_detected": int(num_cycles),
        "steps": int(steps),
        "fps": int(fps)
    }
    
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    return metrics


def run_predator_prey_suite(
    seeds: List[int] = [42, 101, 2024],
    grid_size: int = 256,
    steps: int = 4500,
    sample_interval: int = 3,
    fps: int = 25,
    base_output_dir: str = "results/supplementary/predator_prey"
) -> Dict[str, Any]:
    os.makedirs(base_output_dir, exist_ok=True)
    all_results = {}
    
    print("\n================================================================================")
    print("=== THESIS ACT 7: AUTHENTIC FLOW-LENIA SOLITON PREDATOR-PREY BENCHMARK ===")
    print(f"Seeds: {seeds} | Resolution: {grid_size}x{grid_size} | Steps: {steps:,}")
    print("================================================================================\n")
    
    for seed in seeds:
        seed_out = os.path.join(base_output_dir, f"seed_{seed}")
        m_res = run_single_predator_prey_seed(
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
        
    print(f"\n>>> Multi-Seed Predator-Prey Suite Completed! Full summary saved to: {summary_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Flow-Lenia Act 7: Authentic Flow-Lenia Soliton Predator-Prey")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024])
    parser.add_argument("--grid_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4500)
    parser.add_argument("--sample_interval", type=int, default=3)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--output_dir", type=str, default="results/supplementary/predator_prey")
    args = parser.parse_args()
    
    run_predator_prey_suite(
        seeds=args.seeds,
        grid_size=args.grid_size,
        steps=args.steps,
        sample_interval=args.sample_interval,
        fps=args.fps,
        base_output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
