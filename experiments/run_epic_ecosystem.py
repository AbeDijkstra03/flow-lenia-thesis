#!/usr/bin/env python3
"""
Thesis Experiment 5 (Grand Synthesis): The Living Colosseum Ecosystem.

Unifies all foundational and mechanical discoveries of the thesis:
1. 8 Distinct Biological Species Lineages with multi-shell concentric Gaussian kernels.
2. Spacious Multi-Chamber Colosseum Arena (Central Open Arch + 4 Satellite Navigation Pillars).
3. 4 Dynamic Chemotactic Nutrient Sanctuaries with active foraging attraction (chi = 6.0).
4. Cyclic Resource Grazing & Regeneration (organisms deplete local feeding grounds and migrate through gates).
5. Stochastic Gene-Wise Gumbel-Max mixing with spatial mutation pressure (T_mut = 25).
6. High-Speed Fourier JAX Solver running 22,500 continuous steps (5-minute broadcast HD video).
7. Exhaustive parameter dumps in metadata.json for 100% bit-for-bit scientific reproducibility.
"""

import os
import time
import argparse
import json
import datetime
import numpy as np
import imageio
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Any, List, Tuple

from core.flow_lenia_jax import (
    FlowLeniaParams, FlowLeniaState, precompute_kernel_ffts,
    initialize_multi_patch_state, compute_sobel_gradients,
    moroz_reintegration_tracking, stochastic_gene_wise_mixing
)
from core.visualization import render_dual_panel_rgb, save_motion_heatmap
from core.metrics import evaluate_run_metrics

def create_colosseum_arena_mask(H: int = 384, W: int = 384, gate_phase: int = 0) -> jnp.ndarray:
    """
    Create a spacious, obstacle-rich oceanic arena with 98.9% open fluid area:
    - 1 Central sleek ring arch (radius 24 to 28 px) with dynamic seasonal cardinal archways.
    - Gate Phase 0 & 2: All 4 cardinal archways open.
    - Gate Phase 1: North & South archways open, East & West archways closed.
    - Gate Phase 3: East & West archways open, North & South archways closed.
    - 4 Satellite circular navigation pillars (radius 10 px) at (H//2 +- 70, W//2 +- 70)
    """
    mask = np.ones((H, W), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    cy, cx = H // 2, W // 2
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    theta = np.arctan2(yy - cy, xx - cx)
    
    ring = (r >= 24.0) & (r <= 28.0)
    gate_ang = 32.0 / 26.0
    
    is_ew = (np.abs(theta) < gate_ang / 4.0) | (np.abs(np.abs(theta) - np.pi) < gate_ang / 4.0)
    is_ns = (np.abs(theta - np.pi / 2.0) < gate_ang / 4.0) | (np.abs(theta + np.pi / 2.0) < gate_ang / 4.0)
    
    if gate_phase == 0 or gate_phase == 2:
        is_gate = is_ew | is_ns
    elif gate_phase == 1:
        is_gate = is_ns
    else: # gate_phase == 3
        is_gate = is_ew
        
    mask[ring & (~is_gate)] = 0.0
    
    for dy, dx in [(-70, -70), (-70, 70), (70, -70), (70, 70)]:
        p_dist = np.sqrt((yy - (cy + dy))**2 + (xx - (cx + dx))**2)
        mask[p_dist <= 10.0] = 0.0
        
    return jnp.array(mask, dtype=jnp.float32)

def enforce_no_penetration_walls(vx: jnp.ndarray, vy: jnp.ndarray, w_mask: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply no-penetration boundary conditions (v · n = 0) at solid obstacle interfaces."""
    wall_r = jnp.roll(w_mask, shift=-1, axis=1)
    wall_l = jnp.roll(w_mask, shift=1, axis=1)
    wall_d = jnp.roll(w_mask, shift=-1, axis=0)
    wall_u = jnp.roll(w_mask, shift=1, axis=0)
    
    vx = jnp.where((vx > 0) & (wall_r < 0.5), 0.0, vx)
    vx = jnp.where((vx < 0) & (wall_l < 0.5), 0.0, vx)
    vy = jnp.where((vy > 0) & (wall_d < 0.5), 0.0, vy)
    vy = jnp.where((vy < 0) & (wall_u < 0.5), 0.0, vy)
    return vx * w_mask, vy * w_mask

def create_initial_sanctuary_resource_map(H: int = 384, W: int = 384) -> jnp.ndarray:
    """
    Create 4 rich nutrient foraging sanctuaries in the 4 quadrant chambers.
    """
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    cy, cx = H // 2, W // 2
    s_nw = np.exp(-((yy - (cy - 95))**2 + (xx - (cx - 95))**2) / (2.0 * 40.0**2))
    s_ne = np.exp(-((yy - (cy - 95))**2 + (xx - (cx + 95))**2) / (2.0 * 40.0**2))
    s_sw = np.exp(-((yy - (cy + 95))**2 + (xx - (cx - 95))**2) / (2.0 * 40.0**2))
    s_se = np.exp(-((yy - (cy + 95))**2 + (xx - (cx + 95))**2) / (2.0 * 40.0**2))
    res = 0.15 + 0.85 * np.clip(s_nw + s_ne + s_sw + s_se, 0.0, 1.0)
    return jnp.array(res, dtype=jnp.float32)

def run_single_epic_seed(
    seed: int,
    grid_size: int = 384,
    total_steps: int = 22500,
    sample_interval: int = 3,
    n_patches: int = 8,
    fps: int = 20,
    chemotaxis_chi: float = 6.0,
    base_output_dir: str = "results/epic_ecosystem"
) -> Dict[str, Any]:
    seed_dir = os.path.join(base_output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    
    H, W = grid_size, grid_size
    K = 9
    total_frames = total_steps // sample_interval
    video_duration_sec = total_frames / fps
    
    print(f"\n======================================================================")
    print(f"🎬 GRAND SYNTHESIS ECOSYSTEM (Seed {seed}) - CHEMOTACTIC COLOSSEUM 🎬")
    print(f"======================================================================")
    print(f"Canvas Resolution   : {H}x{W} (Dual Panel: {H*2}x{W})")
    print(f"Simulation Horizon  : {total_steps:,} continuous time steps")
    print(f"Sampling Frequency  : Every {sample_interval} steps -> {total_frames:,} video frames")
    print(f"Target Video Length : {video_duration_sec:.1f} seconds ({video_duration_sec/60:.2f} minutes)")
    print(f"Species Lineages    : {n_patches} interacting genomes")
    print(f"Arena Architecture  : Seasonal Colosseum + 4 Dynamic Chemotactic Sanctuaries (chi = {chemotaxis_chi})")
    print(f"Physics Parameters  : v_scale = 9.0 | alpha = 0.055 | mut_interval = 25")
    print(f"======================================================================")
    
    rng_key = random.PRNGKey(seed)
    
    # Pre-generate 4 seasonal arena masks
    wall_masks = [create_colosseum_arena_mask(H, W, gate_phase=p) for p in range(4)]
    initial_resources = create_initial_sanctuary_resource_map(H, W)
    
    # 1. Precompute concentric ring kernels
    rng_key, subk = random.split(rng_key)
    radii = jnp.sort(random.uniform(subk, (K,), minval=6.0, maxval=15.0))
    kernel_ffts = precompute_kernel_ffts(radii, H, W)
    
    # 2. Configure 8 distinct species genomes
    rng_key, k_mu, k_sigma = random.split(rng_key, 3)
    mu_presets = random.uniform(k_mu, (n_patches, K), minval=0.138, maxval=0.162)
    sigma_presets = random.uniform(k_sigma, (n_patches, K), minval=0.013, maxval=0.018)
    weights_preset = jnp.full((K,), 1.0 / K)
    
    v_scale = 9.0
    alpha_diff = 0.050
    mutation_interval = 25
    depletion_rate = 0.004
    regen_rate = 0.001
    
    # 3. Seed active species with orbital momentum
    print(f"[1/4] Seeding {n_patches} active species with orbital momentum...")
    rng_key, subk_init = random.split(rng_key)
    init_state = initialize_multi_patch_state(
        subk_init, H, W, C=1, K=K, n_patches=n_patches, kernel_radii=radii,
        mu_presets=mu_presets, sigma_presets=sigma_presets, wall_mask=wall_masks[0]
    )
    state = FlowLeniaState(
        init_state.mass, init_state.mu_map, init_state.sigma_map,
        init_state.weights_map, initial_resources, init_state.genome_id_map
    )
    
    # 4. Open streaming MP4 writer
    video_path = os.path.join(seed_dir, "rollout.mp4")
    print(f"[2/4] Initializing H.264 video stream: {video_path}")
    writer = imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", "18", "-preset", "fast"]
    )
    
    chunk_steps = 1200
    num_chunks = total_steps // chunk_steps
    chunk_sample_interval = sample_interval
    
    milestone_frames = []
    milestone_indices = np.linspace(0, total_steps, 12, dtype=int)
    milestone_set = set(milestone_indices.tolist())
    
    step_cursor = 0
    start_time = time.time()
    all_mass_frames = []
    
    def _epic_step_fn(curr_st, key, do_mutation, current_wall_mask):
        mass_primary = curr_st.mass[0] * current_wall_mask
        curr_res = curr_st.resource_map
        
        # Dynamic resource depletion & regeneration
        occupied = mass_primary > 0.10
        new_res = jnp.where(
            occupied,
            jnp.maximum(0.10, curr_res - depletion_rate),
            jnp.minimum(1.00, curr_res + regen_rate)
        )
        
        # Scent gradient with dynamic inversion: exhausted patches push herds to recovering quadrants
        effective_res = jnp.where(new_res < 0.28, 0.56 - new_res, new_res)
        rx, ry = compute_sobel_gradients(effective_res)
        
        # Convolutions
        fft_m = jnp.fft.rfft2(mass_primary)
        U_stack = jax.vmap(lambda k_fft: jnp.fft.irfft2(fft_m * k_fft, s=(H, W)))(kernel_ffts)
        
        G_k = 2.0 * jnp.exp(-((U_stack - curr_st.mu_map)**2) / (2.0 * jnp.square(curr_st.sigma_map) + 1e-8)) - 1.0
        G = jnp.sum(curr_st.weights_map * G_k, axis=0) * (0.6 + 0.4 * new_res)
        
        gx, gy = compute_sobel_gradients(G)
        ax, ay = compute_sobel_gradients(mass_primary)
        
        # Total velocity: physical advection + chemotactic pull to food
        vx = v_scale * ((1.0 - alpha_diff) * gx - alpha_diff * ax) + chemotaxis_chi * rx
        vy = v_scale * ((1.0 - alpha_diff) * gy - alpha_diff * ay) + chemotaxis_chi * ry
        vx = jnp.tanh(vx)
        vy = jnp.tanh(vy)
        vx, vy = enforce_no_penetration_walls(vx, vy, current_wall_mask)
        
        new_mass_primary, ret_c, f_l, f_r, f_u, f_d = moroz_reintegration_tracking(mass_primary, vx, vy)
        new_mass_primary = new_mass_primary * current_wall_mask
        new_mass = curr_st.mass.at[0].set(new_mass_primary)
        
        # Gumbel-Max Gene-Wise Mixing
        key, subk1, subk2, subk3, subk4 = random.split(key, 5)
        new_mu = stochastic_gene_wise_mixing(subk1, curr_st.mu_map, ret_c, f_l, f_r, f_u, f_d)
        new_sig = stochastic_gene_wise_mixing(subk2, curr_st.sigma_map, ret_c, f_l, f_r, f_u, f_d)
        new_w = stochastic_gene_wise_mixing(subk3, curr_st.weights_map, ret_c, f_l, f_r, f_u, f_d)
        gid_2d = curr_st.genome_id_map[None, :, :]
        new_gid = stochastic_gene_wise_mixing(subk4, gid_2d, ret_c, f_l, f_r, f_u, f_d)[0]
        
        # Periodic Localized Mutation
        key, mk1, mk2, mk3 = random.split(key, 4)
        my = random.randint(mk1, (), 0, H)
        mx = random.randint(mk2, (), 0, W)
        yy_m, xx_m = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
        p_mask = (((yy_m - my)**2 + (xx_m - mx)**2) <= 100)[None, :, :]
        mu_n = random.normal(mk3, shape=new_mu.shape) * 0.010
        mutated_mu = jnp.where(p_mask, jnp.clip(new_mu + mu_n, 0.10, 0.25), new_mu)
        new_mu = jnp.where(do_mutation, mutated_mu, new_mu)
            
        return FlowLeniaState(new_mass, new_mu, new_sig, new_w, new_res, new_gid)

        
    wall_masks_jnp = jnp.stack(wall_masks, axis=0) # Shape: (4, H, W)
    
    print(f"\n[3/4] Launching GPU Simulation ({num_chunks} chunks x {chunk_steps} steps)...")
    for chunk_idx in range(num_chunks):
        chunk_start_t = time.time()
        
        # Scan over chunk
        def _scan_chunk(c, step_i):
            curr_s, k = c
            k, sk = random.split(k)
            glob_step = step_cursor + step_i
            do_mut = (glob_step % mutation_interval) == 0
            gate_p = (glob_step // 2500) % 4
            curr_w = wall_masks_jnp[gate_p]
            nxt = _epic_step_fn(curr_s, sk, do_mut, curr_w)
            return (nxt, k), (nxt.mass[0], nxt.genome_id_map)
            
        (state, rng_key), (mass_chunk, gid_chunk) = jax.lax.scan(
            _scan_chunk, (state, rng_key), jnp.arange(chunk_steps)
        )
        
        mass_chunk_np = np.array(mass_chunk)[::chunk_sample_interval]
        gid_chunk_np = np.array(gid_chunk)[::chunk_sample_interval]
        
        for f_idx in range(mass_chunk_np.shape[0]):
            current_step = step_cursor + (f_idx * chunk_sample_interval)
            m_frame = mass_chunk_np[f_idx]
            g_frame = gid_chunk_np[f_idx]
            gate_p = (current_step // 2500) % 4
            
            rgb = render_dual_panel_rgb(m_frame, g_frame, n_patches=n_patches, wall_mask=wall_masks[gate_p])
            writer.append_data(rgb)
            
            if f_idx % 4 == 0:
                all_mass_frames.append(m_frame)
                
            if current_step in milestone_set:
                milestone_frames.append((current_step, rgb))
                
        step_cursor += chunk_steps
        elapsed = time.time() - start_time
        chunk_fps = chunk_steps / (time.time() - chunk_start_t)
        pct = (chunk_idx + 1) / num_chunks * 100
        frames_written = (chunk_idx + 1) * (chunk_steps // chunk_sample_interval)
        video_sec = frames_written / fps
        print(f"  Chunk {chunk_idx+1:02d}/{num_chunks} ({pct:5.1f}%) | Steps: {step_cursor:5d}/{total_steps} | Video: {frames_written:5d}/{total_frames} frames ({video_sec:5.1f}s) | Speed: {chunk_fps:.0f} steps/s")
        
    writer.close()
    total_compute_time = time.time() - start_time
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"\n[4/4] Video complete ({file_size_mb:.2f} MB in {total_compute_time:.2f}s)!")
    
    # 5. Composite Filmstrip
    if len(milestone_frames) > 0:
        n_ms = len(milestone_frames)
        cols = min(6, n_ms)
        rows = int(np.ceil(n_ms / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), facecolor='#05050e')
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        elif cols == 1:
            axes = axes[:, np.newaxis]
            
        for idx, (s_num, rgb_img) in enumerate(milestone_frames):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]
            ax.imshow(rgb_img)
            pct = (s_num / total_steps) * 100.0
            ax.set_title(f"Step {s_num:,} ({pct:.0f}%)", color='white', fontsize=10, pad=4)
            ax.axis('off')
            
        for idx in range(len(milestone_frames), rows * cols):
            r = idx // cols
            c = idx % cols
            axes[r, c].axis('off')
            
        plt.tight_layout()
        filmstrip_path = os.path.join(seed_dir, "trajectory_filmstrip.png")
        plt.savefig(filmstrip_path, dpi=200, facecolor='#05050e', edgecolor='none')
        plt.close()
        
    # 6. Motion Heatmap
    if len(all_mass_frames) > 0:
        heatmap_path = os.path.join(seed_dir, "motion_heatmap.png")
        save_motion_heatmap(np.array(all_mass_frames), heatmap_path)
        
    # 7. Exhaustive Metadata Logging
    summary = {
        "experiment_name": "Epic Multi-Species Chemotactic Colosseum Ecosystem",
        "timestamp_iso": datetime.datetime.now().isoformat(),
        "seed": seed,
        "grid_size": grid_size,
        "total_steps": total_steps,
        "sample_interval": sample_interval,
        "total_frames": total_frames,
        "n_patches": n_patches,
        "fps": fps,
        "video_duration_sec": video_duration_sec,
        "video_file_size_mb": file_size_mb,
        "compute_time_sec": total_compute_time,
        "physics_engine": "JAX FFT Moroz (2020) Semilagrangian Advection",
        "arena_architecture": "Colosseum + 4 Dynamic Chemotactic Foraging Sanctuaries",
        "chemotaxis_chi": chemotaxis_chi,
        "depletion_rate": depletion_rate,
        "regen_rate": regen_rate,
        "mixing_rule": "gene_wise_gumbel_max",
        "mutation_interval": mutation_interval,
        "v_scale": v_scale,
        "alpha_diffusion": alpha_diff,
        "kernel_count_K": K,
        "kernel_radii": [float(r) for r in np.array(radii)],
        "weights": [float(w) for w in np.array(weights_preset)],
        "species_mu_matrix": [[float(val) for val in row] for row in np.array(mu_presets)],
        "species_sigma_matrix": [[float(val) for val in row] for row in np.array(sigma_presets)]
    }
    with open(os.path.join(seed_dir, "metadata.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary

def main():
    parser = argparse.ArgumentParser(description="Epic Chemotactic Ecosystem Multi-Seed Runner")
    parser.add_argument("--grid_size", type=int, default=384, help="Canvas resolution (default: 384)")
    parser.add_argument("--steps", type=int, default=22500, help="Simulation steps (22500 = 5 min @ 20 FPS)")
    parser.add_argument("--sample_interval", type=int, default=3, help="Sampling interval")
    parser.add_argument("--patches", type=int, default=8, help="Species count")
    parser.add_argument("--fps", type=int, default=20, help="Video framerate")
    parser.add_argument("--chi", type=float, default=6.0, help="Chemotaxis pull to food sanctuaries")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024], help="Random seeds to simulate")
    parser.add_argument("--output_dir", type=str, default="results/epic_ecosystem", help="Output directory")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    multiseed_summaries = []
    for s in args.seeds:
        summary = run_single_epic_seed(
            seed=s,
            grid_size=args.grid_size,
            total_steps=args.steps,
            sample_interval=args.sample_interval,
            n_patches=args.patches,
            fps=args.fps,
            chemotaxis_chi=args.chi,
            base_output_dir=args.output_dir
        )
        multiseed_summaries.append(summary)
        
    with open(os.path.join(args.output_dir, "multiseed_summary.json"), "w") as f:
        json.dump(multiseed_summaries, f, indent=2)
        
    print(f"\n======================================================================")
    print(f"Epic Chemotactic Colosseum simulation completed for seeds {args.seeds}!")
    print(f"Aggregated summary written to: {args.output_dir}/multiseed_summary.json")
    print(f"==================================================================")

if __name__ == "__main__":
    main()
