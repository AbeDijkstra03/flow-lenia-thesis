#!/usr/bin/env python3
"""
Thesis Act 6: Topological Navigation, Dynamic Adaptation & Decentralized Transport Networks.

Evaluates how continuous, mass-conserving Flow-Lenia solitonic systems solve complex
spatial graph and transport problems via localized fluid physics and harmonic/geodesic potential fields:

1. Scenario 'maze': Continuous Fluid Maze Solver & Dead-End Pruning.
   - Unitary full-bodied droplet navigating shortest path; evacuating dead-end cul-de-sacs via surface tension.
2. Scenario 'dynamic_reroute': Dynamic Obstacle & Self-Healing Bottleneck Rerouting.
   - Mid-flight gate closure at t=600; dynamic wall mask tracking; backpressure hydrodynamic U-turn and secondary path selection.
3. Scenario 'tokyo_rail': Multi-Terminal Network Synthesis (Tokyo Rail Archipelago).
   - Proliferative branching connecting 4 distributed city hubs through an 8-island archipelago.
4. Scenario 'swarm_channeling': Dual-Droplet Swarm Rendezvous & Bottleneck Fusion.
   - Multi-agent convergence through a narrow funnel into a unified super-droplet.

ARCHITECTURAL NOTE — Hybrid Flow-Lenia + External Geodesic Potential:
  This experiment uses *authentic* Flow-Lenia physics for morphological cohesion:
    - FFT multi-shell ring-kernel convolution
    - Canonical growth mapping G(U)
    - Moroz bilinear reintegration advection (mass-conserving)
    - No-penetration wall boundary conditions

  Navigation (WHICH direction to travel) is driven by an **externally precomputed geodesic
  distance potential** solved via Dijkstra-like wavefront propagation through the obstacle mask.
  This potential is NOT part of the Flow-Lenia PDE.

  The velocity equation is:
    v = tanh( v_scale * [(1-α)∇G - α∇U] + χ * ∇geodesic_potential )

  Scientific rationale: The Flow-Lenia soliton provides self-organizing morphological cohesion and
  mass conservation. The external geodesic gradient provides the navigational objective. This is
  analogous to how biological amoebae combine internal cytoskeletal self-organisation with external
  chemical gradient sensing (chemotaxis) for directed migration through complex terrains.
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Any, List, Optional, Tuple

from core.flow_lenia_jax import (
    FlowLeniaParams, FlowLeniaState, precompute_kernel_ffts,
    compute_sobel_gradients, moroz_reintegration_tracking
)
from core.environment import (
    create_topological_maze_mask,
    create_dynamic_rerouting_mask,
    create_multi_terminal_city_mask,
    create_swarm_funnel_mask,
    solve_geodesic_corridor_potential
)
from core.metrics import evaluate_run_metrics, compute_center_of_mass
from core.visualization import save_experiment_artifacts


def enforce_no_penetration_walls(vx: jnp.ndarray, vy: jnp.ndarray, w_mask: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Apply no-penetration fluid boundary conditions (v · n = 0) at solid obstacle interfaces:
    Zeroes out velocity components pointing directly into adjacent wall pixels to guarantee zero mass leakage.
    """
    wall_r = jnp.roll(w_mask, shift=-1, axis=1)
    wall_l = jnp.roll(w_mask, shift=1, axis=1)
    wall_d = jnp.roll(w_mask, shift=-1, axis=0)
    wall_u = jnp.roll(w_mask, shift=1, axis=0)
    
    vx = jnp.where((vx > 0) & (wall_r < 0.5), 0.0, vx)
    vx = jnp.where((vx < 0) & (wall_l < 0.5), 0.0, vx)
    vy = jnp.where((vy > 0) & (wall_d < 0.5), 0.0, vy)
    vy = jnp.where((vy < 0) & (wall_u < 0.5), 0.0, vy)
    return vx * w_mask, vy * w_mask


def run_scenario_maze(
    seed: int = 42,
    grid_size: int = 256,
    steps: int = 4500,
    sample_interval: int = 3,
    output_dir: str = "results/supplementary/topological_transport/scenario_maze"
) -> Dict[str, Any]:
    """Scenario 1: The Continuous Fluid Maze Solver & Dead-End Pruning."""
    os.makedirs(output_dir, exist_ok=True)
    H, W = grid_size, grid_size
    K = 9
    radii = jnp.linspace(6.0, 15.0, K)
    kernel_ffts = precompute_kernel_ffts(radii, H, W)
    key = random.PRNGKey(seed)
    
    # 1. Create Maze Wall Mask
    wall_mask_np = np.array(create_topological_maze_mask(H, W), dtype=np.float32)
    wall_mask_jnp = jnp.array(wall_mask_np, dtype=jnp.float32)
    
    # 2. Target Goal in Chamber 2 (Right)
    sy, sx = H / 256.0, W / 256.0
    goal_mask_np = np.zeros((H, W), dtype=np.float32)
    y1, y2 = int(round(100 * sy)), int(round(156 * sy))
    x1, x2 = int(round(210 * sx)), int(round(240 * sx))
    goal_mask_np[y1:y2, x1:x2] = 1.0
    
    # 3. Solve Geodesic Distance Potential Field
    print("  [Maze] Solving geodesic distance gradient through labyrinth corridors...")
    pot_np = solve_geodesic_corridor_potential(wall_mask_np, goal_mask_np)
    pot_jnp = jnp.array(pot_np, dtype=jnp.float32)
    
    rx_raw, ry_raw = compute_sobel_gradients(pot_jnp)
    mag = jnp.sqrt(rx_raw**2 + ry_raw**2)
    rx = jnp.where(mag > 1e-5, rx_raw / (mag + 1e-4), 0.0) * wall_mask_jnp
    ry = jnp.where(mag > 1e-5, ry_raw / (mag + 1e-4), 0.0) * wall_mask_jnp
    
    # 4. Seed-Dependent Stochastic Initialization & Parameters
    k1, k2, k3, k4 = random.split(key, 4)
    jit_y = float(random.uniform(k1, minval=-2.0, maxval=2.0))
    jit_x = float(random.uniform(k2, minval=-2.0, maxval=2.0))
    angle = float(random.uniform(k3, minval=0.0, maxval=2.0 * np.pi))
    kx_dir, ky_dir = np.cos(angle), np.sin(angle)
    
    chi = 0.28
    alpha_val = 0.040
    mu_seed = 0.150 + float(random.normal(k4)) * 0.003
    sigma_val = 0.016
    v_s = 7.0
    
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    cy, cx = int(round(128 * sy)) + jit_y, int(round(35 * sx)) + jit_x
    r_dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    slope = 1.0 + 0.35 * (kx_dir * (xx - cx) + ky_dir * (yy - cy)) / (15.0 * sx)
    blob = np.exp(-r_dist**2 / (2.0 * (10.0 * sy)**2)) * np.clip(slope, 0.5, 1.5)
    init_mass_2d = np.where(r_dist < (15.0 * sy), np.clip(blob, 0.0, 1.0), 0.0) * wall_mask_np
    initial_mass_total = float(np.sum(init_mass_2d))
    
    init_state = FlowLeniaState(
        jnp.array(init_mass_2d[None, :, :], dtype=jnp.float32),
        jnp.full((K, H, W), mu_seed, dtype=jnp.float32),
        jnp.full((K, H, W), sigma_val, dtype=jnp.float32),
        jnp.full((K, H, W), 1.0 / K, dtype=jnp.float32),
        pot_jnp,
        jnp.zeros((H, W), dtype=jnp.int32)
    )
    
    @jax.jit
    def _step_fn(curr_st):
        mass_p = curr_st.mass[0] * wall_mask_jnp
        fft_m = jnp.fft.rfft2(mass_p)
        U_stack = jax.vmap(lambda k_fft: jnp.fft.irfft2(fft_m * k_fft, s=(H, W)))(kernel_ffts)
        G_k = 2.0 * jnp.exp(-((U_stack - curr_st.mu_map)**2) / (2.0 * jnp.square(curr_st.sigma_map) + 1e-8)) - 1.0
        G = jnp.sum(curr_st.weights_map * G_k, axis=0)
        gx, gy = compute_sobel_gradients(G)
        ax, ay = compute_sobel_gradients(mass_p)
        vx = v_s * ((1.0 - alpha_val) * gx - alpha_val * ax) + chi * rx
        vy = v_s * ((1.0 - alpha_val) * gy - alpha_val * ay) + chi * ry
        vx = jnp.tanh(vx)
        vy = jnp.tanh(vy)
        vx, vy = enforce_no_penetration_walls(vx, vy, wall_mask_jnp)
        new_mass_p, _, _, _, _, _ = moroz_reintegration_tracking(mass_p, vx, vy)
        new_mass_p = new_mass_p * wall_mask_jnp
        new_mass = curr_st.mass.at[0].set(new_mass_p)
        return curr_st._replace(mass=new_mass)
    
    # 5. Execute Rollout
    print(f"  [Maze] Running {steps} steps rollout (Seed {seed})...")
    curr_st = init_state
    sampled_frames = []
    dead_end_1_masses = []
    dead_end_2_masses = []
    goal_masses = []
    
    dead_1_box = (yy >= 14*sy) & (yy <= 55*sy) & (xx >= 140*sx) & (xx <= 168*sx)
    dead_2_box = (yy >= 115*sy) & (yy <= 140*sy) & (xx >= 100*sx) & (xx <= 138*sx)
    goal_box = (xx >= 200*sx) & (yy >= 90*sy) & (yy <= 166*sy)
    
    for t in range(steps):
        curr_st = _step_fn(curr_st)
        if t % sample_interval == 0:
            m_np = np.array(curr_st.mass[0])
            sampled_frames.append(m_np)
            dead_end_1_masses.append(float(np.sum(m_np[dead_1_box])))
            dead_end_2_masses.append(float(np.sum(m_np[dead_2_box])))
            goal_masses.append(float(np.sum(m_np[goal_box])))
            
    sampled_frames_arr = np.array(sampled_frames)
    final_mass = float(np.sum(sampled_frames_arr[-1]))
    final_goal_mass = float(goal_masses[-1])
    success_reach_goal = (final_goal_mass > 0.30 * initial_mass_total)
    
    max_dead_1 = float(np.max(dead_end_1_masses)) if len(dead_end_1_masses) > 0 else 0.0
    end_dead_1 = float(dead_end_1_masses[-1]) if len(dead_end_1_masses) > 0 else 0.0
    dead_1_evacuated = (max_dead_1 > 5.0 and end_dead_1 < 2.0)
    
    preservation = float(final_mass / (initial_mass_total + 1e-8))
    print(f"  [Maze Result] Goal Reached: {success_reach_goal} | Mass in Goal: {final_goal_mass:.1f}/{initial_mass_total:.1f} | Preservation: {preservation*100:.1f}%")
    
    config = {
        "scenario": "continuous_maze",
        "grid_size": grid_size,
        "steps": steps,
        "sample_interval": sample_interval,
        "seed": seed,
        "chi": chi,
        "alpha_diffusion": alpha_val,
        "v_scale": v_s
    }
    
    metrics = {
        "scenario": "continuous_maze",
        "seed": seed,
        "success_reach_goal": bool(success_reach_goal),
        "initial_mass": initial_mass_total,
        "final_mass": final_mass,
        "final_goal_mass": final_goal_mass,
        "mass_preservation_ratio": preservation,
        "max_dead_end_1_mass": max_dead_1,
        "final_dead_end_1_mass": end_dead_1,
        "dead_end_1_evacuated": bool(dead_1_evacuated),
        "steps": steps
    }
    
    gid_maps = np.zeros_like(sampled_frames_arr, dtype=np.int32)
    save_experiment_artifacts(
        sampled_mass_frames=sampled_frames_arr,
        metrics=metrics,
        config=config,
        output_dir=output_dir,
        fps=30,
        wall_mask=wall_mask_jnp,
        genome_id_maps=gid_maps,
        use_subfolder=False
    )
    
    return metrics


def run_scenario_dynamic_reroute(
    seed: int = 42,
    grid_size: int = 256,
    steps: int = 4500,
    switch_step: int = 600,
    sample_interval: int = 3,
    output_dir: str = "results/supplementary/topological_transport/scenario_dynamic_reroute"
) -> Dict[str, Any]:
    """Scenario 2: Dynamic Obstacle & Self-Healing Bottleneck Rerouting."""
    os.makedirs(output_dir, exist_ok=True)
    H, W = grid_size, grid_size
    K = 9
    radii = jnp.linspace(6.0, 15.0, K)
    kernel_ffts = precompute_kernel_ffts(radii, H, W)
    key = random.PRNGKey(seed)
    
    # 1. Create Masks for Open Phase and Closed Phase
    wall_open_np = np.array(create_dynamic_rerouting_mask(H, W, gate_closed=False), dtype=np.float32)
    wall_closed_np = np.array(create_dynamic_rerouting_mask(H, W, gate_closed=True), dtype=np.float32)
    
    wall_open_jnp = jnp.array(wall_open_np, dtype=jnp.float32)
    wall_closed_jnp = jnp.array(wall_closed_np, dtype=jnp.float32)
    
    # 2. Goal in Right Chamber
    sy, sx = H / 256.0, W / 256.0
    goal_mask_np = np.zeros((H, W), dtype=np.float32)
    y1, y2 = int(round(100 * sy)), int(round(155 * sy))
    x1, x2 = int(round(215 * sx)), int(round(238 * sx))
    goal_mask_np[y1:y2, x1:x2] = 1.0
    
    # 3. Solve Geodesic Potentials for Both Open and Closed Topologies
    print("  [Dynamic Reroute] Solving geodesic distance potentials for Open and Closed gate topologies...")
    pot_open_np = solve_geodesic_corridor_potential(wall_open_np, goal_mask_np)
    pot_closed_np = solve_geodesic_corridor_potential(wall_closed_np, goal_mask_np)
    
    rx_raw, ry_raw = compute_sobel_gradients(jnp.array(pot_open_np))
    mag_o = jnp.sqrt(rx_raw**2 + ry_raw**2)
    rx_open = jnp.where(mag_o > 1e-5, rx_raw / (mag_o + 1e-4), 0.0) * wall_open_jnp
    ry_open = jnp.where(mag_o > 1e-5, ry_raw / (mag_o + 1e-4), 0.0) * wall_open_jnp
    
    rx_raw, ry_raw = compute_sobel_gradients(jnp.array(pot_closed_np))
    mag_c = jnp.sqrt(rx_raw**2 + ry_raw**2)
    rx_closed = jnp.where(mag_c > 1e-5, rx_raw / (mag_c + 1e-4), 0.0) * wall_closed_jnp
    ry_closed = jnp.where(mag_c > 1e-5, ry_raw / (mag_c + 1e-4), 0.0) * wall_closed_jnp
    
    # 4. Seed-Dependent Stochastic Initialization & Parameters
    k1, k2, k3, k4 = random.split(key, 4)
    jit_y = float(random.uniform(k1, minval=-2.0, maxval=2.0))
    jit_x = float(random.uniform(k2, minval=-2.0, maxval=2.0))
    angle = float(random.uniform(k3, minval=0.0, maxval=2.0 * np.pi))
    kx_dir, ky_dir = np.cos(angle), np.sin(angle)
    
    chi = 0.28
    alpha_val = 0.040
    mu_seed = 0.150 + float(random.normal(k4)) * 0.003
    sigma_val = 0.016
    v_s = 7.0
    
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    cy, cx = int(round(128 * sy)) + jit_y, int(round(35 * sx)) + jit_x
    r_dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    slope = 1.0 + 0.35 * (kx_dir * (xx - cx) + ky_dir * (yy - cy)) / (15.0 * sx)
    blob = np.exp(-r_dist**2 / (2.0 * (10.0 * sy)**2)) * np.clip(slope, 0.5, 1.5)
    init_mass_2d = np.where(r_dist < (15.0 * sy), np.clip(blob, 0.0, 1.0), 0.0) * wall_open_np
    initial_mass_total = float(np.sum(init_mass_2d))
    
    init_state = FlowLeniaState(
        jnp.array(init_mass_2d[None, :, :], dtype=jnp.float32),
        jnp.full((K, H, W), mu_seed, dtype=jnp.float32),
        jnp.full((K, H, W), sigma_val, dtype=jnp.float32),
        jnp.full((K, H, W), 1.0 / K, dtype=jnp.float32),
        jnp.array(pot_open_np),
        jnp.zeros((H, W), dtype=jnp.int32)
    )
    
    @jax.jit
    def _step_fn(curr_st, current_wall_mask, current_rx, current_ry):
        mass_p = curr_st.mass[0] * current_wall_mask
        fft_m = jnp.fft.rfft2(mass_p)
        U_stack = jax.vmap(lambda k_fft: jnp.fft.irfft2(fft_m * k_fft, s=(H, W)))(kernel_ffts)
        G_k = 2.0 * jnp.exp(-((U_stack - curr_st.mu_map)**2) / (2.0 * jnp.square(curr_st.sigma_map) + 1e-8)) - 1.0
        G = jnp.sum(curr_st.weights_map * G_k, axis=0)
        gx, gy = compute_sobel_gradients(G)
        ax, ay = compute_sobel_gradients(mass_p)
        vx = v_s * ((1.0 - alpha_val) * gx - alpha_val * ax) + chi * current_rx
        vy = v_s * ((1.0 - alpha_val) * gy - alpha_val * ay) + chi * current_ry
        vx = jnp.tanh(vx)
        vy = jnp.tanh(vy)
        vx, vy = enforce_no_penetration_walls(vx, vy, current_wall_mask)
        new_mass_p, _, _, _, _, _ = moroz_reintegration_tracking(mass_p, vx, vy)
        new_mass_p = new_mass_p * current_wall_mask
        new_mass = curr_st.mass.at[0].set(new_mass_p)
        return curr_st._replace(mass=new_mass)
    
    print(f"  [Dynamic Reroute] Executing rollout. Gate will abruptly close at Step t={switch_step} (Seed {seed})...")
    curr_st = init_state
    sampled_frames = []
    wall_mask_frames = []
    north_masses = []
    south_masses = []
    goal_masses = []
    
    north_box = (yy >= 50*sy) & (yy <= 80*sy) & (xx >= 80*sx) & (xx <= 160*sx)
    south_box = (yy >= 175*sy) & (yy <= 205*sy) & (xx >= 80*sx) & (xx <= 160*sx)
    goal_box = (xx >= 200*sx) & (yy >= 85*sy) & (yy <= 170*sy)
    
    for t in range(steps):
        if t < switch_step:
            w_mask, c_rx, c_ry = wall_open_jnp, rx_open, ry_open
            current_w_np = wall_open_np
        else:
            w_mask, c_rx, c_ry = wall_closed_jnp, rx_closed, ry_closed
            current_w_np = wall_closed_np
            
        curr_st = _step_fn(curr_st, w_mask, c_rx, c_ry)
        
        if t % sample_interval == 0:
            m_np = np.array(curr_st.mass[0])
            sampled_frames.append(m_np)
            wall_mask_frames.append(current_w_np)
            north_masses.append(float(np.sum(m_np[north_box])))
            south_masses.append(float(np.sum(m_np[south_box])))
            goal_masses.append(float(np.sum(m_np[goal_box])))
            
    sampled_frames_arr = np.array(sampled_frames)
    wall_mask_frames_arr = np.array(wall_mask_frames)
    final_mass = float(np.sum(sampled_frames_arr[-1]))
    final_goal_mass = float(goal_masses[-1])
    
    max_north_mass = float(np.max(north_masses)) if len(north_masses) > 0 else 0.0
    final_north_mass = float(north_masses[-1]) if len(north_masses) > 0 else 0.0
    final_south_mass = float(south_masses[-1]) if len(south_masses) > 0 else 0.0
    
    rerouting_success = (final_mass > 0.50 * initial_mass_total) and (final_north_mass < 5.0)
    preservation = float(final_mass / (initial_mass_total + 1e-8))
    print(f"  [Dynamic Result] Reroute Success: {rerouting_success} | Final Total Mass: {final_mass:.1f}/{initial_mass_total:.1f} | Preservation: {preservation*100:.1f}%")
    
    config = {
        "scenario": "dynamic_rerouting",
        "grid_size": grid_size,
        "steps": steps,
        "switch_step": switch_step,
        "sample_interval": sample_interval,
        "seed": seed,
        "chi": chi,
        "alpha_diffusion": alpha_val
    }
    
    metrics = {
        "scenario": "dynamic_rerouting",
        "seed": seed,
        "switch_step": switch_step,
        "rerouting_success": bool(rerouting_success),
        "initial_mass": initial_mass_total,
        "final_mass": final_mass,
        "final_goal_mass": final_goal_mass,
        "mass_preservation_ratio": preservation,
        "max_north_mass": max_north_mass,
        "final_north_mass": final_north_mass,
        "steps": steps
    }
    
    gid_maps = np.zeros_like(sampled_frames_arr, dtype=np.int32)
    save_experiment_artifacts(
        sampled_mass_frames=sampled_frames_arr,
        metrics=metrics,
        config=config,
        output_dir=output_dir,
        fps=30,
        wall_mask=wall_mask_frames_arr,
        genome_id_maps=gid_maps,
        use_subfolder=False
    )
    
    return metrics


def run_scenario_tokyo_rail(
    seed: int = 42,
    grid_size: int = 256,
    steps: int = 4500,
    sample_interval: int = 3,
    output_dir: str = "results/supplementary/topological_transport/scenario_tokyo_rail"
) -> Dict[str, Any]:
    """Scenario 3: Multi-Terminal Network Synthesis (The Tokyo Rail Archipelago Experiment)."""
    os.makedirs(output_dir, exist_ok=True)
    H, W = grid_size, grid_size
    K = 9
    radii = jnp.linspace(6.0, 15.0, K)
    kernel_ffts = precompute_kernel_ffts(radii, H, W)
    key = random.PRNGKey(seed)
    
    wall_mask_np = np.array(create_multi_terminal_city_mask(H, W), dtype=np.float32)
    wall_mask_jnp = jnp.array(wall_mask_np, dtype=jnp.float32)
    
    sy, sx = H / 256.0, W / 256.0
    cities = [
        ("City_NW", int(round(45 * sy)), int(round(45 * sx))),
        ("City_NE", int(round(45 * sy)), int(round(211 * sx))),
        ("City_SE", int(round(211 * sy)), int(round(211 * sx))),
        ("City_SW", int(round(211 * sy)), int(round(45 * sx)))
    ]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    goal_mask_np = np.zeros((H, W), dtype=np.float32)
    city_boxes = {}
    
    for c_name, cy, cx in cities:
        r_c = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        goal_mask_np[r_c <= (18.0 * sy)] = 1.0
        city_boxes[c_name] = (r_c <= (22.0 * sy))
        
    print("  [Tokyo Rail] Solving multi-terminal geodesic potential around archipelago mountain passes...")
    pot_np = solve_geodesic_corridor_potential(wall_mask_np, goal_mask_np)
    pot_jnp = jnp.array(pot_np, dtype=jnp.float32)
    
    rx_raw, ry_raw = compute_sobel_gradients(pot_jnp)
    mag = jnp.sqrt(rx_raw**2 + ry_raw**2)
    rx_t = jnp.where(mag > 1e-5, rx_raw / (mag + 1e-4), 0.0) * wall_mask_jnp
    ry_t = jnp.where(mag > 1e-5, ry_raw / (mag + 1e-4), 0.0) * wall_mask_jnp
    
    k1, k2 = random.split(key, 2)
    chi = 0.26
    alpha_val = 0.038
    mu_seed = 0.150 + float(random.normal(k1)) * 0.002
    sigma_val = 0.015
    v_s = 7.5
    
    cy, cx = int(round(128 * sy)), int(round(128 * sx))
    r_dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    blob = np.exp(-r_dist**2 / (2.0 * (18.0 * sy)**2))
    init_mass_2d = np.where(r_dist < (26.0 * sy), np.clip(blob, 0.0, 1.0), 0.0) * wall_mask_np
    initial_mass_total = float(np.sum(init_mass_2d))
    
    init_state = FlowLeniaState(
        jnp.array(init_mass_2d[None, :, :], dtype=jnp.float32),
        jnp.full((K, H, W), mu_seed, dtype=jnp.float32),
        jnp.full((K, H, W), sigma_val, dtype=jnp.float32),
        jnp.full((K, H, W), 1.0 / K, dtype=jnp.float32),
        pot_jnp,
        jnp.zeros((H, W), dtype=jnp.int32)
    )
    
    @jax.jit
    def _step_fn(curr_st):
        mass_p = curr_st.mass[0] * wall_mask_jnp
        fft_m = jnp.fft.rfft2(mass_p)
        U_stack = jax.vmap(lambda k_fft: jnp.fft.irfft2(fft_m * k_fft, s=(H, W)))(kernel_ffts)
        G_k = 2.0 * jnp.exp(-((U_stack - curr_st.mu_map)**2) / (2.0 * jnp.square(curr_st.sigma_map) + 1e-8)) - 1.0
        G = jnp.sum(curr_st.weights_map * G_k, axis=0)
        gx, gy = compute_sobel_gradients(G)
        ax, ay = compute_sobel_gradients(mass_p)
        vx = v_s * ((1.0 - alpha_val) * gx - alpha_val * ax) + chi * rx_t
        vy = v_s * ((1.0 - alpha_val) * gy - alpha_val * ay) + chi * ry_t
        vx = jnp.tanh(vx)
        vy = jnp.tanh(vy)
        vx, vy = enforce_no_penetration_walls(vx, vy, wall_mask_jnp)
        new_mass_p, _, _, _, _, _ = moroz_reintegration_tracking(mass_p, vx, vy)
        new_mass_p = new_mass_p * wall_mask_jnp
        new_mass = curr_st.mass.at[0].set(new_mass_p)
        return curr_st._replace(mass=new_mass)
    
    print(f"  [Tokyo Rail] Running archipelago network synthesis across {steps} steps (Seed {seed})...")
    curr_st = init_state
    sampled_frames = []
    city_mass_histories = {c: [] for c in city_boxes}
    
    for t in range(steps):
        curr_st = _step_fn(curr_st)
        if t % sample_interval == 0:
            m_np = np.array(curr_st.mass[0])
            sampled_frames.append(m_np)
            for c_name, box in city_boxes.items():
                city_mass_histories[c_name].append(float(np.sum(m_np[box])))
                
    sampled_frames_arr = np.array(sampled_frames)
    final_mass = float(np.sum(sampled_frames_arr[-1]))
    
    city_final_masses = {c: city_mass_histories[c][-1] for c in city_boxes}
    all_cities_connected = all(m > 25.0 for m in city_final_masses.values()) if len(city_final_masses) > 0 else False
    preservation = float(final_mass / (initial_mass_total + 1e-8))
    
    print(f"  [Tokyo Rail Result] All Cities Connected: {all_cities_connected} | Final Masses: {city_final_masses} | Preservation: {preservation*100:.1f}%")
    
    config = {
        "scenario": "tokyo_rail",
        "grid_size": grid_size,
        "steps": steps,
        "sample_interval": sample_interval,
        "seed": seed,
        "chi": chi,
        "alpha_diffusion": alpha_val
    }
    
    metrics = {
        "scenario": "tokyo_rail",
        "seed": seed,
        "all_cities_connected": bool(all_cities_connected),
        "city_final_masses": city_final_masses,
        "initial_mass": initial_mass_total,
        "final_mass": final_mass,
        "mass_preservation_ratio": preservation,
        "steps": steps
    }
    
    gid_maps = np.zeros_like(sampled_frames_arr, dtype=np.int32)
    save_experiment_artifacts(
        sampled_mass_frames=sampled_frames_arr,
        metrics=metrics,
        config=config,
        output_dir=output_dir,
        fps=30,
        wall_mask=wall_mask_jnp,
        genome_id_maps=gid_maps,
        use_subfolder=False
    )
    
    return metrics


def run_scenario_swarm_channeling(
    seed: int = 42,
    grid_size: int = 256,
    steps: int = 4500,
    sample_interval: int = 3,
    output_dir: str = "results/supplementary/topological_transport/scenario_swarm_channeling"
) -> Dict[str, Any]:
    """Scenario 4 (Bonus): Dual-Droplet Swarm Rendezvous & Narrow Bottleneck Fusion."""
    os.makedirs(output_dir, exist_ok=True)
    H, W = grid_size, grid_size
    K = 9
    radii = jnp.linspace(6.0, 15.0, K)
    kernel_ffts = precompute_kernel_ffts(radii, H, W)
    key = random.PRNGKey(seed)
    
    wall_mask_np = np.array(create_swarm_funnel_mask(H, W, bottleneck_width=32), dtype=np.float32)
    wall_mask_jnp = jnp.array(wall_mask_np, dtype=jnp.float32)
    
    sy, sx = H / 256.0, W / 256.0
    goal_mask_np = np.zeros((H, W), dtype=np.float32)
    y1, y2 = int(round(100 * sy)), int(round(156 * sy))
    x1, x2 = int(round(210 * sx)), int(round(240 * sx))
    goal_mask_np[y1:y2, x1:x2] = 1.0
    
    print("  [Swarm Channeling] Solving geodesic potential inside converging funnel...")
    pot_np = solve_geodesic_corridor_potential(wall_mask_np, goal_mask_np)
    pot_jnp = jnp.array(pot_np, dtype=jnp.float32)
    
    rx_raw, ry_raw = compute_sobel_gradients(pot_jnp)
    mag = jnp.sqrt(rx_raw**2 + ry_raw**2)
    rx_s = jnp.where(mag > 1e-5, rx_raw / (mag + 1e-4), 0.0) * wall_mask_jnp
    ry_s = jnp.where(mag > 1e-5, ry_raw / (mag + 1e-4), 0.0) * wall_mask_jnp
    
    k1, k2, k3 = random.split(key, 3)
    chi = 0.28
    alpha_val = 0.040
    mu_seed = 0.150 + float(random.normal(k1)) * 0.003
    sigma_val = 0.016
    v_s = 7.0
    
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    jit1 = float(random.uniform(k2, minval=-2.0, maxval=2.0))
    jit2 = float(random.uniform(k3, minval=-2.0, maxval=2.0))
    
    r1 = np.sqrt((yy - int(round(70*sy)) - jit1)**2 + (xx - int(round(45*sx)))**2)
    r2 = np.sqrt((yy - int(round(185*sy)) - jit2)**2 + (xx - int(round(45*sx)))**2)
    blob1 = np.exp(-r1**2 / (2.0 * (10.0*sy)**2))
    blob2 = np.exp(-r2**2 / (2.0 * (10.0*sy)**2))
    m1 = np.where(r1 < (15.0*sy), np.clip(blob1, 0.0, 1.0), 0.0)
    m2 = np.where(r2 < (15.0*sy), np.clip(blob2, 0.0, 1.0), 0.0)
    
    init_mass_2d = (m1 + m2) * wall_mask_np
    initial_mass_total = float(np.sum(init_mass_2d))
    
    init_state = FlowLeniaState(
        jnp.array(init_mass_2d[None, :, :], dtype=jnp.float32),
        jnp.full((K, H, W), mu_seed, dtype=jnp.float32),
        jnp.full((K, H, W), sigma_val, dtype=jnp.float32),
        jnp.full((K, H, W), 1.0 / K, dtype=jnp.float32),
        pot_jnp,
        jnp.zeros((H, W), dtype=jnp.int32)
    )
    
    @jax.jit
    def _step_fn(curr_st):
        mass_p = curr_st.mass[0] * wall_mask_jnp
        fft_m = jnp.fft.rfft2(mass_p)
        U_stack = jax.vmap(lambda k_fft: jnp.fft.irfft2(fft_m * k_fft, s=(H, W)))(kernel_ffts)
        G_k = 2.0 * jnp.exp(-((U_stack - curr_st.mu_map)**2) / (2.0 * jnp.square(curr_st.sigma_map) + 1e-8)) - 1.0
        G = jnp.sum(curr_st.weights_map * G_k, axis=0)
        gx, gy = compute_sobel_gradients(G)
        ax, ay = compute_sobel_gradients(mass_p)
        vx = v_s * ((1.0 - alpha_val) * gx - alpha_val * ax) + chi * rx_s
        vy = v_s * ((1.0 - alpha_val) * gy - alpha_val * ay) + chi * ry_s
        vx = jnp.tanh(vx)
        vy = jnp.tanh(vy)
        vx, vy = enforce_no_penetration_walls(vx, vy, wall_mask_jnp)
        new_mass_p, _, _, _, _, _ = moroz_reintegration_tracking(mass_p, vx, vy)
        new_mass_p = new_mass_p * wall_mask_jnp
        new_mass = curr_st.mass.at[0].set(new_mass_p)
        return curr_st._replace(mass=new_mass)
    
    print(f"  [Swarm Channeling] Running dual-droplet rendezvous rollout across {steps} steps (Seed {seed})...")
    curr_st = init_state
    sampled_frames = []
    east_chamber_box = (xx >= 150*sx) & (yy >= 45*sy) & (yy <= 210*sy)
    east_masses = []
    
    for t in range(steps):
        curr_st = _step_fn(curr_st)
        if t % sample_interval == 0:
            m_np = np.array(curr_st.mass[0])
            sampled_frames.append(m_np)
            east_masses.append(float(np.sum(m_np[east_chamber_box])))
            
    sampled_frames_arr = np.array(sampled_frames)
    final_mass = float(np.sum(sampled_frames_arr[-1]))
    final_east_mass = float(east_masses[-1])
    
    successful_fusion = (final_east_mass > 0.30 * initial_mass_total)
    preservation = float(final_mass / (initial_mass_total + 1e-8))
    print(f"  [Swarm Result] Successful Fusion & Passage: {successful_fusion} | Final East Mass: {final_east_mass:.1f}/{initial_mass_total:.1f} | Preservation: {preservation*100:.1f}%")
    
    config = {
        "scenario": "swarm_channeling",
        "grid_size": grid_size,
        "steps": steps,
        "sample_interval": sample_interval,
        "seed": seed,
        "chi": chi,
        "alpha_diffusion": alpha_val
    }
    
    metrics = {
        "scenario": "swarm_channeling",
        "seed": seed,
        "successful_fusion": bool(successful_fusion),
        "initial_mass": initial_mass_total,
        "final_mass": final_mass,
        "final_east_mass": final_east_mass,
        "mass_preservation_ratio": preservation,
        "steps": steps
    }
    
    gid_maps = np.zeros_like(sampled_frames_arr, dtype=np.int32)
    save_experiment_artifacts(
        sampled_mass_frames=sampled_frames_arr,
        metrics=metrics,
        config=config,
        output_dir=output_dir,
        fps=30,
        wall_mask=wall_mask_jnp,
        genome_id_maps=gid_maps,
        use_subfolder=False
    )
    
    return metrics


def run_topological_suite(
    seeds: List[int] = [42, 101, 2024],
    scenario: str = "all",
    grid_size: int = 256,
    steps: int = 4500,
    sample_interval: int = 3,
    base_output_dir: str = "results/supplementary/topological_transport"
) -> Dict[str, Any]:
    """Execute the full Topological Transport Suite across specified scenarios and multiple seeds."""
    os.makedirs(base_output_dir, exist_ok=True)
    all_results = {}
    
    print("\n================================================================================")
    print("=== THESIS ACT 6: TOPOLOGICAL NAVIGATION, DYNAMIC REROUTING & SWARM NETWORKS ===")
    print(f"Scenarios: {scenario} | Seeds: {seeds} | Resolution: {grid_size}x{grid_size} | Steps: {steps}")
    print("================================================================================\n")
    
    for seed in seeds:
        seed_res = {}
        
        # Scenario 1: Continuous Maze
        if scenario in ["all", "maze"]:
            print(f"\n--- [Seed {seed}] Scenario 1: Continuous Maze Solver & Dead-End Pruning ---")
            m_res = run_scenario_maze(
                seed=seed,
                grid_size=grid_size,
                steps=steps,
                sample_interval=sample_interval,
                output_dir=os.path.join(base_output_dir, f"scenario_maze/seed_{seed}")
            )
            seed_res["maze"] = m_res
            
        # Scenario 2: Dynamic Obstacle Rerouting
        if scenario in ["all", "dynamic_reroute"]:
            print(f"\n--- [Seed {seed}] Scenario 2: Dynamic Obstacle & Fault-Tolerant Rerouting ---")
            d_res = run_scenario_dynamic_reroute(
                seed=seed,
                grid_size=grid_size,
                steps=steps,
                switch_step=600,
                sample_interval=sample_interval,
                output_dir=os.path.join(base_output_dir, f"scenario_dynamic_reroute/seed_{seed}")
            )
            seed_res["dynamic_reroute"] = d_res
            
        # Scenario 3: Tokyo Rail Multi-Terminal Network
        if scenario in ["all", "tokyo_rail"]:
            print(f"\n--- [Seed {seed}] Scenario 3: Tokyo Rail Multi-Terminal Network Synthesis ---")
            t_res = run_scenario_tokyo_rail(
                seed=seed,
                grid_size=grid_size,
                steps=steps,
                sample_interval=sample_interval,
                output_dir=os.path.join(base_output_dir, f"scenario_tokyo_rail/seed_{seed}")
            )
            seed_res["tokyo_rail"] = t_res
            
        # Scenario 4: Swarm Channeling
        if scenario in ["all", "swarm_channeling"]:
            print(f"\n--- [Seed {seed}] Scenario 4: Dual-Droplet Swarm Rendezvous & Bottleneck Fusion ---")
            s_res = run_scenario_swarm_channeling(
                seed=seed,
                grid_size=grid_size,
                steps=steps,
                sample_interval=sample_interval,
                output_dir=os.path.join(base_output_dir, f"scenario_swarm_channeling/seed_{seed}")
            )
            seed_res["swarm_channeling"] = s_res
            
        all_results[f"seed_{seed}"] = seed_res
        
    summary_path = os.path.join(base_output_dir, "multiseed_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\n>>> Multi-Seed Suite Completed! Full summary saved to: {summary_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Flow-Lenia Act 6: Topological Transport Suite")
    parser.add_argument("--scenario", type=str, choices=["all", "maze", "dynamic_reroute", "tokyo_rail", "swarm_channeling"], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024])
    parser.add_argument("--grid_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=4500)
    parser.add_argument("--sample_interval", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="results/supplementary/topological_transport")
    args = parser.parse_args()
    
    run_topological_suite(
        seeds=args.seeds,
        scenario=args.scenario,
        grid_size=args.grid_size,
        steps=args.steps,
        sample_interval=args.sample_interval,
        base_output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
