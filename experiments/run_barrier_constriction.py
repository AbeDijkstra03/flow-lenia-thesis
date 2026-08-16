#!/usr/bin/env python3
"""
Novel Thesis Experiment: Morphological Plasticity & Corridor Navigation under Spatial Constrictions.

Evaluates how self-organizing Flow-Lenia solitons traverse geometric constrictions across
a parameter sweep of passage widths (e.g. W_p in [8, 16, 24, 32] px).
Computes transmission coefficients T(W), mass preservation ratios, and exports
high-definition MP4 videos, filmstrips, and transmission curve plots for thesis figures.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from typing import List, Dict, Any, Optional

from core.flow_lenia_jax import (
    FlowLeniaParams, FlowLeniaState, precompute_kernel_ffts, initialize_multi_patch_state, run_flow_lenia_rollout
)
from core.environment import create_wall_obstacle_mask
from core.metrics import evaluate_run_metrics, compute_center_of_mass
from core.visualization import save_experiment_artifacts, save_rollout_mp4, extract_trajectory_filmstrip

def create_single_corridor_wall_mask(
    H: int = 256,
    W: int = 256,
    wall_x: Optional[int] = None,
    wall_thickness: int = 8,
    passage_width: int = 20,
    passage_y: Optional[int] = None
) -> jnp.ndarray:
    """
    Create a static wall obstacle mask with a single configurable passage corridor.
    1.0 = passable vacuum, 0.0 = static wall.
    """
    mask = np.ones((H, W), dtype=np.float32)
    if wall_x is None:
        wall_x = W // 2
    if passage_y is None:
        passage_y = H // 2
        
    x_start = max(0, wall_x - wall_thickness // 2)
    x_end = min(W, wall_x + wall_thickness // 2)
    
    # Build vertical wall
    mask[:, x_start:x_end] = 0.0
    
    # Cut single corridor passage
    p_start = max(0, passage_y - passage_width // 2)
    p_end = min(H, passage_y + passage_width // 2)
    mask[p_start:p_end, x_start:x_end] = 1.0
    
    return jnp.array(mask, dtype=jnp.float32)

def run_constriction_sweep(
    passage_widths: List[int] = [8, 16, 24, 32],
    grid_size: int = 256,
    steps: int = 3600,
    sample_interval: int = 3,
    seed: int = 42,
    output_dir: str = "results/barrier_constriction"
):
    os.makedirs(output_dir, exist_ok=True)
    rng_key = random.PRNGKey(seed)
    
    H, W = grid_size, grid_size
    K = 9
    
    print(f"=== Flow-Lenia Corridor Constriction Experiment ===")
    print(f"Grid: {H}x{W} | Steps: {steps} | Passage Widths: {passage_widths} | JAX: {jax.devices()}")
    
    # Fixed tuned glider genome for controlled comparative sweep
    rng_key, subk = random.split(rng_key)
    radii = jnp.sort(random.uniform(subk, (K,), minval=6.0, maxval=15.0))
    kernel_ffts = precompute_kernel_ffts(radii, H, W)
    
    n_patches = 3
    rng_key, k_mu, k_sigma = random.split(rng_key, 3)
    mu_presets = random.uniform(k_mu, (n_patches, K), minval=0.14, maxval=0.18)
    sigma_presets = random.uniform(k_sigma, (n_patches, K), minval=0.012, maxval=0.018)
    
    v_scale = 5.4
    alpha_diff = 0.06
    params = FlowLeniaParams(
        mu=mu_presets[0], sigma=sigma_presets[0], weights=jnp.full((K,), 1.0 / K),
        v_scale=v_scale, alpha_diffusion=alpha_diff
    )
    
    sweep_results = []
    
    for p_width in passage_widths:
        print(f"\n--- Testing Passage Width W = {p_width} px ---")
        wall_mask = create_single_corridor_wall_mask(
            H=H, W=W, wall_x=W // 2, wall_thickness=8, passage_width=p_width, passage_y=H // 2
        )
        
        # Initialize organism in left chamber (x ~ W * 0.25)
        rng_key, subk_init = random.split(rng_key)
        state = initialize_multi_patch_state(
            subk_init, H, W, C=1, K=K, n_patches=n_patches, kernel_radii=radii,
            mu_presets=mu_presets, sigma_presets=sigma_presets, wall_mask=wall_mask
        )
        
        rng_key, subk_rollout = random.split(rng_key)
        final_state, sampled_mass, sampled_gid = run_flow_lenia_rollout(
            state, kernel_ffts, params, subk_rollout,
            num_steps=steps,
            sample_interval=sample_interval,
            wall_mask=wall_mask,
            enable_mutation=False
        )
        
        sampled_mass_np = np.array(sampled_mass)
        final_mass = sampled_mass_np[-1, 0] if sampled_mass_np.ndim == 4 else sampled_mass_np[-1]
        init_mass = sampled_mass_np[0, 0] if sampled_mass_np.ndim == 4 else sampled_mass_np[0]
        
        # Quantitative Metrics:
        # 1. Total Mass Conservation
        init_tot = float(np.sum(init_mass)) + 1e-8
        final_tot = float(np.sum(final_mass)) + 1e-8
        mass_preservation = final_tot / init_tot
        
        # 2. Transmission Coefficient T = Mass in Right Chamber (x > W//2) / Total Mass
        right_chamber_mass = float(np.sum(final_mass[:, W // 2:]))
        transmission_coeff = right_chamber_mass / final_tot
        
        # 3. Core Integrity (Solid core fraction)
        core_mask = (final_mass >= 0.15)
        solid_core_ratio = float(np.sum(final_mass * core_mask)) / final_tot
        
        # 4. Center of Mass Displacement
        cy0, cx0 = compute_center_of_mass(init_mass)
        cy_end, cx_end = compute_center_of_mass(final_mass)
        com_shift = float(np.sqrt((cy_end - cy0)**2 + (cx_end - cx0)**2))
        
        print(f"Transmission Coeff T: {transmission_coeff:.4f} ({transmission_coeff*100:.1f}%)")
        print(f"Mass Preservation: {mass_preservation:.6f} | Solid Core Ratio: {solid_core_ratio:.4f} | CoM Shift: {com_shift:.2f} px")
        
        prefix = f"constriction_W{p_width}"
        config_dict = {
            "passage_width": p_width,
            "grid_size": grid_size,
            "steps": steps,
            "v_scale": v_scale,
            "alpha_diffusion": alpha_diff
        }
        metric_dict = {
            "passage_width": p_width,
            "transmission_coeff": transmission_coeff,
            "mass_preservation": mass_preservation,
            "solid_core_ratio": solid_core_ratio,
            "com_shift": com_shift,
            "right_chamber_mass": right_chamber_mass,
            "total_mass": final_tot
        }
        
        art_paths = save_experiment_artifacts(
            sampled_mass_frames=sampled_mass_np,
            metrics=metric_dict,
            config=config_dict,
            output_dir=output_dir,
            prefix=prefix,
            fps=20,
            wall_mask=np.array(wall_mask)
        )
        
        sweep_results.append(metric_dict)
        
    # Generate Publication-Ready Transmission Curve Plot
    plt.figure(figsize=(7, 5), dpi=300)
    widths = [r["passage_width"] for r in sweep_results]
    t_coeffs = [r["transmission_coeff"] * 100.0 for r in sweep_results]
    core_ratios = [r["solid_core_ratio"] * 100.0 for r in sweep_results]
    
    plt.plot(widths, t_coeffs, 'o-', color='#2b5c8f', linewidth=2.5, markersize=8, label='Chamber Transmission $T(W)$ (%)')
    plt.plot(widths, core_ratios, 's--', color='#e05a47', linewidth=2.0, markersize=7, label='Solid Core Preservation (%)')
    
    plt.title("Flow-Lenia Soliton Transmission vs. Corridor Passage Width", fontsize=12, fontweight='bold')
    plt.xlabel("Corridor Passage Width $W_{\\text{passage}}$ (pixels)", fontsize=11)
    plt.ylabel("Percentage (%)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.xticks(widths)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "transmission_curve.png")
    plt.savefig(plot_path)
    plt.close()
    
    summary_path = os.path.join(output_dir, "constriction_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
        
    print(f"\n=== Constriction Sweep Complete ===")
    print(f"Transmission Curve Plot: {plot_path}")
    print(f"Full Summary JSON: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Corridor Constriction Thesis Experiment")
    parser.add_argument("--widths", nargs="+", type=int, default=[8, 16, 24, 32], help="Passage widths (px)")
    parser.add_argument("--grid_size", type=int, default=256, help="Grid size resolution")
    parser.add_argument("--steps", type=int, default=3600, help="Simulation steps (default: 3600 -> 1 min video)")
    parser.add_argument("--sample_interval", type=int, default=3, help="Sampling interval (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results/barrier_constriction", help="Output directory")
    args = parser.parse_args()
    
    run_constriction_sweep(
        passage_widths=args.widths,
        grid_size=args.grid_size,
        steps=args.steps,
        sample_interval=args.sample_interval,
        seed=args.seed,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
