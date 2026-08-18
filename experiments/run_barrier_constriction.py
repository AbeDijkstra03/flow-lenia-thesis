#!/usr/bin/env python3
"""
Thesis Experiment 4B: Soft-Bodied Barrier Constriction & Chemotactic Foraging.

Evaluates how soft-bodied continuous Flow-Lenia organisms navigate geometric constrictions
under directed chemotactic nutrient attraction toward a food source placed in Chamber 2.
Organism is configured with high surface tension (alpha=0.065, sigma=0.013, chi=18.0) to maintain a cohesive, solid elastomeric single-core droplet.
Sweeps passage widths W_p in [8, 16, 24, 32, 48, 64] px across multiple random seeds.
"""

import os
import json
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from typing import List, Dict, Any, Optional

from core.flow_lenia_jax import (
    FlowLeniaParams, FlowLeniaState, precompute_kernel_ffts,
    compute_sobel_gradients, moroz_reintegration_tracking
)
from core.metrics import evaluate_run_metrics, compute_center_of_mass
from core.visualization import save_experiment_artifacts

def create_single_corridor_wall_mask(
    H: int = 256,
    W: int = 256,
    wall_x: Optional[int] = None,
    wall_thickness: int = 8,
    passage_width: int = 20,
    passage_y: Optional[int] = None
) -> jnp.ndarray:
    """
    Create a static barrier wall separating Chamber 1 (left) and Chamber 2 (right)
    with a single configurable corridor passage aperture.
    1.0 = passable fluid domain, 0.0 = static solid wall.
    """
    mask = np.ones((H, W), dtype=np.float32)
    if wall_x is None:
        wall_x = W // 2
    if passage_y is None:
        passage_y = H // 2
        
    x_start = max(0, wall_x - wall_thickness // 2)
    x_end = min(W, wall_x + wall_thickness // 2)
    
    mask[:, x_start:x_end] = 0.0
    
    p_start = max(0, passage_y - passage_width // 2)
    p_end = min(H, passage_y + passage_width // 2)
    mask[p_start:p_end, x_start:x_end] = 1.0
    
    return jnp.array(mask, dtype=jnp.float32)

def run_single_constriction_seed(
    seed: int,
    passage_widths: List[int] = [8, 16, 24, 32, 48, 64],
    grid_size: int = 256,
    steps: int = 3600,
    sample_interval: int = 3,
    chemotaxis_chi: float = 18.0,
    base_output_dir: str = "results/barrier_constriction"
) -> Dict[str, Any]:
    seed_dir = os.path.join(base_output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    rng_key = random.PRNGKey(seed)
    
    H, W = grid_size, grid_size
    K = 9
    radii = jnp.linspace(6.0, 15.0, K)
    kernel_ffts = precompute_kernel_ffts(radii, H, W)
    
    # 1. Setup Nutrient / Food Bait Field: 0.0 in Chamber 1, ramping smoothly through slit to 1.0 in Chamber 2
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    resource_map_np = np.clip((xx - 65.0) / 75.0, 0.0, 1.0).astype(np.float32)
    resource_jnp = jnp.array(resource_map_np, dtype=jnp.float32)
    rx, ry = compute_sobel_gradients(resource_jnp)
    
    mu_presets = jnp.full((1, K), 0.150)
    sigma_presets = jnp.full((1, K), 0.013)
    weights_preset = jnp.full((K,), 1.0 / K)
    
    v_scale = 8.5
    alpha_diff = 0.065
    
    params = FlowLeniaParams(
        mu=mu_presets[0],
        sigma=sigma_presets[0],
        weights=weights_preset,
        v_scale=v_scale,
        alpha_diffusion=alpha_diff
    )
    
    print(f"\n==================================================================")
    print(f"=== Chemotactic Constriction Sweep (Seed {seed}) ===")
    print(f"Chamber 1 (x < 128): Spawn Zone | Chamber 2 (x > 128): Nutrient Pool")
    print(f"Cohesive Single-Core (alpha=0.065, sigma=0.013) | Pull (chi) = {chemotaxis_chi}")
    print(f"==================================================================")
    
    sweep_results = []
    
    def _step_with_chemotaxis(state: FlowLeniaState, wall_mask: jnp.ndarray) -> FlowLeniaState:
        mass_primary = state.mass[0] * wall_mask
        fft_m = jnp.fft.rfft2(mass_primary)
        def _conv_k(k_fft):
            return jnp.fft.irfft2(fft_m * k_fft, s=(H, W))
        U_stack = jax.vmap(_conv_k)(kernel_ffts)
        G_k = 2.0 * jnp.exp(-((U_stack - state.mu_map)**2) / (2.0 * jnp.square(state.sigma_map) + 1e-8)) - 1.0
        G = jnp.sum(state.weights_map * G_k, axis=0)
        gx, gy = compute_sobel_gradients(G)
        ax, ay = compute_sobel_gradients(mass_primary)
        
        vx = v_scale * ((1.0 - alpha_diff) * gx - alpha_diff * ax) + chemotaxis_chi * rx
        vy = v_scale * ((1.0 - alpha_diff) * gy - alpha_diff * ay) + chemotaxis_chi * ry
        vx = jnp.tanh(vx) * wall_mask
        vy = jnp.tanh(vy) * wall_mask
        
        new_mass_primary, _, _, _, _, _ = moroz_reintegration_tracking(mass_primary, vx, vy)
        new_mass_primary = new_mass_primary * wall_mask
        return FlowLeniaState(
            new_mass_primary[None, :, :],
            state.mu_map,
            state.sigma_map,
            state.weights_map,
            state.resource_map,
            state.genome_id_map
        )
    
    for p_width in passage_widths:
        print(f"\n--- [Seed {seed}] Testing Passage Width W = {p_width:2d} px ---")
        wall_mask = create_single_corridor_wall_mask(
            H=H, W=W, wall_x=W // 2, wall_thickness=8, passage_width=p_width, passage_y=H // 2
        )
        
        # Place organism cleanly in Chamber 1 at cy=128, cx=65
        cy, cx = 128, 65
        r_dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
        blob = np.exp(-r_dist**2 / (2.0 * 14.0**2))
        init_mass_2d = np.where(r_dist < 22.0, np.clip(blob, 0.0, 1.0), 0.0) * np.array(wall_mask)
        initial_total_mass = float(np.sum(init_mass_2d))
        
        init_state = FlowLeniaState(
            jnp.array(init_mass_2d[None, :, :], dtype=jnp.float32),
            jnp.full((K, H, W), 0.150, dtype=jnp.float32),
            jnp.full((K, H, W), 0.013, dtype=jnp.float32),
            jnp.full((K, H, W), 1.0 / K, dtype=jnp.float32),
            resource_jnp,
            jnp.zeros((H, W), dtype=jnp.int32)
        )
        
        def _scan_fn(curr_st, step_idx):
            next_st = _step_with_chemotaxis(curr_st, wall_mask)
            return next_st, next_st.mass[0]
            
        final_state, all_mass_frames = jax.lax.scan(_scan_fn, init_state, jnp.arange(steps))
        
        sampled_indices = np.arange(0, steps, sample_interval)
        sampled_mass_np = np.array(all_mass_frames)[sampled_indices]
        sampled_mass_np = sampled_mass_np[:, None, :, :]
        
        final_mass_arr = sampled_mass_np[-1, 0]
        sampled_gid_np = np.zeros((sampled_mass_np.shape[0], H, W), dtype=np.int32)
        
        metrics = evaluate_run_metrics(final_mass_arr, sampled_mass_np, sampled_gid_np, steps)
        
        # Calculate chamber 2 mass transmission ratio T(W)
        c2_mask = np.zeros((H, W), dtype=bool)
        c2_mask[:, (W // 2) + 4:] = True
        
        final_c2_mass = float(np.sum(final_mass_arr[c2_mask]))
        transmission_ratio = float(final_c2_mass / max(initial_total_mass, 1e-8))
        
        cy_init, cx_init = compute_center_of_mass(init_mass_2d)
        cy_final, cx_final = compute_center_of_mass(final_mass_arr)
        com_dx = float(cx_final - cx_init)
        
        print(f"Transmission T(W)         : {transmission_ratio * 100:.1f}% ({final_c2_mass:.1f}/{initial_total_mass:.1f} mass units)")
        print(f"Solid Core Preservation   : {metrics['solid_core_ratio'] * 100:.1f}%")
        print(f"Horizontal CoM Shift (dx) : {com_dx:.2f} px")
        
        config_dict = {
            "experiment": "Chemotactic Corridor Constriction",
            "passage_width": p_width,
            "grid_size": grid_size,
            "steps": steps,
            "seed": seed,
            "chemotaxis_chi": chemotaxis_chi,
            "transmission_ratio": transmission_ratio,
            "com_dx": com_dx,
            "cohesion": "high_surface_tension"
        }
        
        prefix = f"width_{p_width:02d}"
        art_paths = save_experiment_artifacts(
            sampled_mass_frames=sampled_mass_np,
            metrics=metrics,
            config=config_dict,
            output_dir=seed_dir,
            prefix=prefix,
            fps=20,
            wall_mask=wall_mask
        )
        
        sweep_results.append({
            "passage_width": p_width,
            "transmission_ratio": transmission_ratio,
            "chamber2_mass": final_c2_mass,
            "initial_mass": initial_total_mass,
            "solid_core_ratio": float(metrics["solid_core_ratio"]),
            "mass_preservation": float(metrics["mass_preservation_ratio"]),
            "com_dx": com_dx,
            "metrics": metrics,
            "artifacts": art_paths
        })
        
    widths = [r["passage_width"] for r in sweep_results]
    transmissions = [r["transmission_ratio"] * 100.0 for r in sweep_results]
    solidity = [r["solid_core_ratio"] * 100.0 for r in sweep_results]
    
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(widths, transmissions, marker='o', linewidth=3, markersize=8, color='#185ADB', label=r'Chamber Transmission $T(W)$ (%)')
    plt.plot(widths, solidity, marker='s', linewidth=2.5, linestyle='--', markersize=7, color='#FF5722', label='Solid Core Preservation (%)')
    plt.title(f"Chemotactic Soliton Transmission vs. Corridor Width (Seed {seed})", fontsize=13, fontweight='bold')
    plt.xlabel(r"Corridor Passage Width $W_{\mathrm{passage}}$ (pixels)", fontsize=11)
    plt.ylabel("Percentage (%)", fontsize=11)
    plt.xticks(widths)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, loc='center right')
    plt.tight_layout()
    plot_path = os.path.join(seed_dir, "transmission_curve.png")
    plt.savefig(plot_path)
    plt.close()
    
    summary_data = {
        "experiment_name": "Chemotactic Corridor Constriction",
        "timestamp_iso": datetime.datetime.now().isoformat(),
        "seed": seed,
        "chemotaxis_chi": chemotaxis_chi,
        "passage_widths": widths,
        "transmission_ratios": [r["transmission_ratio"] for r in sweep_results],
        "sweep_results": sweep_results
    }
    with open(os.path.join(seed_dir, "constriction_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)
        
    return summary_data

def main():
    parser = argparse.ArgumentParser(description="Chemotactic Corridor Constriction Multi-Seed Runner")
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64], help="Passage widths (px)")
    parser.add_argument("--grid_size", type=int, default=256, help="Grid size")
    parser.add_argument("--steps", type=int, default=3600, help="Simulation steps (3600 = 1 min video)")
    parser.add_argument("--sample_interval", type=int, default=3, help="Sampling interval")
    parser.add_argument("--chi", type=float, default=18.0, help="Chemotaxis attraction strength")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 101, 2024], help="Random seeds to evaluate")
    parser.add_argument("--output_dir", type=str, default="results/barrier_constriction", help="Output directory")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    multiseed_summaries = []
    for s in args.seeds:
        summary = run_single_constriction_seed(
            seed=s,
            passage_widths=args.widths,
            grid_size=args.grid_size,
            steps=args.steps,
            sample_interval=args.sample_interval,
            chemotaxis_chi=args.chi,
            base_output_dir=args.output_dir
        )
        multiseed_summaries.append(summary)
        
    with open(os.path.join(args.output_dir, "multiseed_transmission_summary.json"), "w") as f:
        json.dump(multiseed_summaries, f, indent=2)
        
    print(f"\n==================================================================")
    print(f"Chemotactic Constriction sweep across all {len(args.seeds)} seeds complete!")
    print(f"Aggregated summary written to: {args.output_dir}/multiseed_transmission_summary.json")
    print(f"==================================================================")

if __name__ == "__main__":
    main()
